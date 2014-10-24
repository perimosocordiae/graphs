import numpy as np
from sklearn import linear_model
from graphs import Graph, plot_graph

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
  cv = min(d, 3)
  if positive or d < n:
    clf = linear_model.LassoCV(positive=positive, cv=cv)
  else:
    clf = linear_model.LassoLarsCV(cv=cv)
  # Normalize all samples
  X = X / np.linalg.norm(X, ord=2, axis=1)[:,None]
  # Solve for each row of W
  # TODO: use a sparse vstack to assemble rows
  W = np.zeros((n,n))
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
  return Graph.from_adj_matrix(W)
