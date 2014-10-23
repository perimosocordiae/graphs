import numpy as np
from sklearn import linear_model
from graphs import Graph, plot_graph
from common.util import Progress

__all__ = ['sparse_regularized_graph']

# TODO: implement NNLRS next
# http://www.cis.pku.edu.cn/faculty/vision/zlin/Publications/2012-CVPR-NNLRS.pdf


def sparse_regularized_graph(X, positive=False):
  '''
  Commonly known as an l1-graph.
  When positive=True, known as SPG (sparse probability graph).

  l1-graph: Semi-supervised Learning by Sparse Representation
  Yan & Wang, SDM 2009
  http://epubs.siam.org/doi/pdf/10.1137/1.9781611972795.68

  SPG: Nonnegative Sparse Coding for Discriminative Semi-supervised Learning
  CVPR 2001
  '''
  n,d = X.shape
  I = np.eye(d)
  # Choose an efficient Lasso solver
  if positive or d < n:
    clf = linear_model.LassoCV(positive=positive)
  else:
    clf = linear_model.LassoLarsCV()
  # Normalize all samples
  X = X / np.linalg.norm(X, ord=2, axis=1)[:,None]
  # Solve for each row of W
  # TODO: use a sparse vstack to assemble rows
  W = np.zeros((n,n))
  spinner = Progress(total=n)
  for i in xrange(n):
    x = X[i]
    B = np.vstack((X[:i], X[i+1:], I)).T
    # Solve min ||B'a - x|| + |a|
    clf.fit(B, x)
    a = np.abs(clf.coef_[:n-1])
    a /= a.sum()
    # Assign edges
    W[i,:i] = a[:i]
    W[i,i+1:] = a[i:]
    spinner.update()
  spinner.finish()
  return Graph.from_adj_matrix(W)


def demo():
  from matplotlib.pyplot import subplots, set_cmap
  from common.neighborhood import neighbor_graph
  from common.synthetic_data import gaussian_clusters

  X = gaussian_clusters(5, 100, 20)
  G_l1 = sparse_regularized_graph(X)
  G_spg = sparse_regularized_graph(X, positive=True)
  G_knn = neighbor_graph(X, k=4, weighting='none')

  _, axes = subplots(nrows=2, ncols=3)
  axes[1,0].imshow(G_l1.matrix(), interpolation='nearest')
  axes[1,1].imshow(G_spg.matrix(), interpolation='nearest')
  axes[1,2].imshow(G_knn.matrix(), interpolation='nearest')
  set_cmap('YlOrRd')
  for ax in axes[0]:
    ax.set_axis_bgcolor('gray')
  X2d = X[:,:2]
  plot_graph(G_l1, X2d, ax=axes[0,0], title='L1', unweighted=False)
  plot_graph(G_spg, X2d, ax=axes[0,1], title='SPG', unweighted=False)
  plot_graph(G_knn, X2d, ax=axes[0,2], title='kNN', unweighted=False)()

if __name__ == '__main__':
  demo()
