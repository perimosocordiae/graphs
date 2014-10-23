import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from graphs import Graph, plot_graph

from common.distance import SquaredL2

__all__ = ['perturbed_mst', 'disjoint_mst']


def perturbed_mst(X, num_perturbations=20, metric=SquaredL2,
                  jitter=None, plot=False):
  '''Builds a graph as the union of several MSTs on perturbed data.
  Reference: http://ecovision.mit.edu/~sloop/shao.pdf, page 8
  jitter refers to the scale of the gaussian noise added for each perturbation.
  When jitter is None, it defaults to the 5th percentile interpoint distance.'''
  D = SquaredL2.within(X)
  if jitter is None:
    jitter = np.percentile(D[D>0], 5)
  W = minimum_spanning_tree(D)
  W.data[:] = 1.0  # binarize
  for i in xrange(num_perturbations):
    if plot:
      plot_graph(Graph.from_adj_matrix(W), X, title='%d edges' % W.nnz)()
    pX = X + np.random.normal(scale=jitter, size=X.shape)
    pW = minimum_spanning_tree(SquaredL2.within(pX))
    pW.data[:] = 1.0
    W = W + pW
  # final graph is the average over all pertubed MSTs + the original
  W.data /= (num_perturbations + 1.0)
  return W


def disjoint_mst(X, num_spanning_trees=3, metric=SquaredL2, plot=False):
  '''Builds a graph as the union of several spanning trees,
  each time removing any edges present in previously-built trees.
  Reference: http://ecovision.mit.edu/~sloop/shao.pdf, page 9.'''
  D = SquaredL2.within(X)
  mst = minimum_spanning_tree(D)
  W = mst.copy()
  for i in xrange(1, num_spanning_trees):
    if plot:
      plot_graph(Graph.from_adj_matrix(W), X, title='%d edges' % W.nnz)()
    ii,jj = mst.nonzero()
    D[ii,jj] = np.inf
    D[jj,ii] = np.inf
    mst = minimum_spanning_tree(D)
    W = W + mst
  return W


def _make_animation(X, func, filename):
  from common.viz import FigureSaver
  with FigureSaver(animation=filename):
    W = func(X, plot=True)
    plot_graph(Graph.from_adj_matrix(W), X, title='Final: %d edges' % W.nnz)()
  print 'See results in', filename


if __name__ == '__main__':
  from common.synthetic_data import circle, add_noise
  from matplotlib import pyplot
  X = add_noise(circle(50), 0.1)
  # _make_animation(X, perturbed_mst, 'pmst.gif')
  # _make_animation(X, disjoint_mst, 'dmst.gif')

  # show them side by side
  _, (ax1,ax2) = pyplot.subplots(ncols=2)
  Wd = disjoint_mst(X)
  Wp = perturbed_mst(X)
  plot_graph(Graph.from_adj_matrix(Wp), X, ax=ax1, title='P-MST')
  plot_graph(Graph.from_adj_matrix(Wd), X, ax=ax2, title='D-MST')()
