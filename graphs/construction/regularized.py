from __future__ import absolute_import

import numpy as np
import scipy.sparse as ss
from sklearn import linear_model
from graphs import Graph
from .neighbors import nearest_neighbors

__all__ = ['sparse_regularized_graph']

# TODO: implement NNLRS next
# http://www.cis.pku.edu.cn/faculty/vision/zlin/Publications/2012-CVPR-NNLRS.pdf


def sparse_regularized_graph(X, positive=False, alpha=None, k=None):
  '''
  Commonly known as an l1-graph.
  When positive=True, known as SPG (sparse probability graph).
  When alpha=None, uses cross-validation to find sparsity parameters. This is
  very slow, but it gets good results.
  When k=None, allow all points to be edges. Otherwise, restrict to kNN set.

  l1-graph: Semi-supervised Learning by Sparse Representation
  Yan & Wang, SDM 2009
  http://epubs.siam.org/doi/pdf/10.1137/1.9781611972795.68

  SPG: Nonnegative Sparse Coding for Discriminative Semi-supervised Learning
  He et al., CVPR 2001
  '''
  clf, X = _l1_graph_setup(X, positive, alpha)
  if k is None:
    W = _l1_graph_solve_full(clf, X)
  else:
    W = _l1_graph_solve_k(clf, X, k)
  return Graph.from_adj_matrix(W)


def _l1_graph_solve_full(clf, X):
  n, d = X.shape
  # Solve for each row of W
  W = []
  B = np.vstack((X[1:], np.eye(d)))
  for i, x in enumerate(X):
    # Solve min ||B'a - x|| + |a|
    clf.fit(B.T, x)
    # Set up B for next time
    B[i] = x
    # Extract edge weights (first n-1 coefficients)
    a = ss.csr_matrix(clf.coef_[:n-1])
    a = abs(a)
    a /= a.sum()
    # Add a zero on the diagonal
    a.indices[np.searchsorted(a.indices, i):] += 1
    a._shape = (1, n)  # XXX: hack around lack of csr.resize()
    W.append(a)
  return ss.vstack(W)


def _l1_graph_solve_k(clf, X, k):
  n, d = X.shape
  nn_inds = nearest_neighbors(X, k=k+1)  # self-edges included
  # Solve for each row of W
  W = []
  B = np.empty((k+d, d))
  B[k:] = np.eye(d)
  for i, x in enumerate(X):
    # Set up B with neighbors of x
    idx = nn_inds[i]
    idx = idx[idx!=i]  # remove self-edge
    B[:k] = X[idx]
    # Solve min ||B'a - x|| + |a|
    clf.fit(B.T, x)
    # Extract edge weights (first k coefficients)
    a = ss.csr_matrix((clf.coef_[:k], idx, [0, k]), shape=(1, n))
    a.eliminate_zeros()  # some of the first k might be zeros
    a = abs(a)
    a /= a.sum()
    W.append(a)
  return ss.vstack(W)


def _l1_graph_setup(X, positive, alpha):
  n, d = X.shape
  # Choose an efficient Lasso solver
  if alpha is not None:
    if positive or d < n:
      clf = linear_model.Lasso(positive=positive, alpha=alpha)
    else:
      clf = linear_model.LassoLars(alpha=alpha)
  else:
    cv = min(d, 3)
    if positive or d < n:
      clf = linear_model.LassoCV(positive=positive, cv=cv)
    else:
      clf = linear_model.LassoLarsCV(cv=cv)
  # Normalize all samples
  X = X / np.linalg.norm(X, ord=2, axis=1)[:,None]
  return clf, X
