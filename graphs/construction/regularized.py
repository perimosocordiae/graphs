import numpy as np
import scipy.sparse as ss
from sklearn import linear_model
from graphs import Graph

__all__ = ['sparse_regularized_graph']

# TODO: implement NNLRS next
# http://www.cis.pku.edu.cn/faculty/vision/zlin/Publications/2012-CVPR-NNLRS.pdf


def sparse_regularized_graph(X, positive=False, alpha=None):
  '''
  Commonly known as an l1-graph.
  When positive=True, known as SPG (sparse probability graph).
  When alpha=None, uses cross-validation to find sparsity parameters. This is
  very slow, but it gets good results.

  l1-graph: Semi-supervised Learning by Sparse Representation
  Yan & Wang, SDM 2009
  http://epubs.siam.org/doi/pdf/10.1137/1.9781611972795.68

  SPG: Nonnegative Sparse Coding for Discriminative Semi-supervised Learning
  He et al., CVPR 2001
  '''
  n,d = X.shape
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
    a = np.abs(a)
    a /= a.sum()
    # Add a zero on the diagonal
    a.indices[np.searchsorted(a.indices, i):] += 1
    a._shape = (1, n)  # XXX: hack around lack of csr.resize()
    W.append(a)
  return Graph.from_adj_matrix(ss.vstack(W))
