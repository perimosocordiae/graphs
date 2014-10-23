import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.metrics.pairwise import pairwise_distances
from graphs import Graph, plot_graph

__all__ = ['perturbed_mst', 'disjoint_mst']


def perturbed_mst(X, num_perturbations=20, metric='euclidean',
                  jitter=None, plot=False):
  '''Builds a graph as the union of several MSTs on perturbed data.
  Reference: http://ecovision.mit.edu/~sloop/shao.pdf, page 8
  jitter refers to the scale of the gaussian noise added for each perturbation.
  When jitter is None, it defaults to the 5th percentile interpoint distance.'''
  D = pairwise_distances(X, metric=metric)
  if jitter is None:
    jitter = np.percentile(D[D>0], 5)
  W = minimum_spanning_tree(D)
  W.data[:] = 1.0  # binarize
  for i in xrange(num_perturbations):
    if plot:
      plot_graph(Graph.from_adj_matrix(W), X, title='%d edges' % W.nnz)()
    pX = X + np.random.normal(scale=jitter, size=X.shape)
    pW = minimum_spanning_tree(pairwise_distances(pX, metric=metric))
    pW.data[:] = 1.0
    W = W + pW
  # final graph is the average over all pertubed MSTs + the original
  W.data /= (num_perturbations + 1.0)
  return Graph.from_adj_matrix(W)


def disjoint_mst(X, num_spanning_trees=3, metric='euclidean', plot=False):
  '''Builds a graph as the union of several spanning trees,
  each time removing any edges present in previously-built trees.
  Reference: http://ecovision.mit.edu/~sloop/shao.pdf, page 9.'''
  D = pairwise_distances(X, metric=metric)
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
  return Graph.from_adj_matrix(W)
