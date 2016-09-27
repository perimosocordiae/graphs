from __future__ import absolute_import, division

import numpy as np
import scipy.sparse as ss
import warnings
from scipy.linalg import cho_factor, get_lapack_funcs
from sklearn import linear_model
from sklearn.metrics import pairwise_distances

from graphs import Graph
from ..mini_six import range
from .neighbors import nearest_neighbors

__all__ = ['sparse_regularized_graph', 'smce_graph']

# For quickly running cho_solve without lots of checking
potrs = get_lapack_funcs('potrs')

# TODO: implement NNLRS next
# http://www.cis.pku.edu.cn/faculty/vision/zlin/Publications/2012-CVPR-NNLRS.pdf


def smce_graph(X, metric='l2', sparsity_param=10, kmax=None, keep_ratio=0.95):
  '''Sparse graph construction from the SMCE paper.

  X : 2-dimensional array-like
  metric : str, optional
  sparsity_param : float, optional
  kmax : int, optional
  keep_ratio : float, optional
    When <1, keep edges up to (keep_ratio * total weight)

  Returns a graph with asymmetric similarity weights.
  Call .symmetrize() and .kernelize('rbf') to convert to symmetric distances.

  SMCE: "Sparse Manifold Clustering and Embedding"
    Elhamifar & Vidal, NIPS 2011
  '''
  n = X.shape[0]
  if kmax is None:
    kmax = min(n-1, max(5, n // 10))

  nn_dists, nn_inds = nearest_neighbors(X, metric=metric, k=kmax+1,
                                        return_dists=True)
  W = np.zeros((n, n))

  # optimize each point separately
  for i, pt in enumerate(X):
    nbr_inds = nn_inds[i]
    mask = nbr_inds != i  # remove self-edge
    nbr_inds = nbr_inds[mask]
    nbr_dist = nn_dists[i,mask]
    Y = (X[nbr_inds] - pt) / nbr_dist[:,None]
    # solve sparse optimization with ADMM
    c = _solve_admm(Y, nbr_dist/nbr_dist.sum(), sparsity_param)
    c = np.abs(c / nbr_dist)
    W[i,nbr_inds] = c / c.sum()

  W = ss.csr_matrix(W)
  if keep_ratio < 1:
    for i in range(n):
      row_data = W.data[W.indptr[i]:W.indptr[i+1]]
      order = np.argsort(row_data)[::-1]
      stop_idx = np.searchsorted(np.cumsum(row_data[order]), keep_ratio) + 1
      bad_inds = order[stop_idx:]
      row_data[bad_inds] = 0
    W.eliminate_zeros()

  return Graph.from_adj_matrix(W)


def _solve_admm(Y, q, alpha=10, mu=10, max_iter=10000):
  n = Y.shape[0]
  alpha_q = alpha * q
  # solve (YYt + mu*I + mu) Z = (mu*C - lambda + gamma + mu)
  A, lower = cho_factor(Y.dot(Y.T) + mu*(np.eye(n) + 1), overwrite_a=True)
  C = np.zeros(n)
  Z_old = 0  # shape (n,)
  lmbda = np.zeros(n)
  gamma = 0
  # ADMM iteration
  for i in range(max_iter):
    # call the guts of cho_solve directly for speed
    Z, _ = potrs(A, gamma + mu + mu*C - lmbda, lower=lower, overwrite_b=True)

    tmp = mu*Z + lmbda
    C[:] = np.abs(tmp)
    C -= alpha_q
    np.maximum(C, 0, out=C)
    C *= np.sign(tmp)
    C /= mu

    d_ZC = Z - C
    d_1Z = 1 - Z.sum()
    lmbda += mu * d_ZC
    gamma += mu * d_1Z

    if ((abs(d_1Z) / n < 1e-6)
            and (np.abs(d_ZC).mean() < 1e-6)
            and (np.abs(Z - Z_old).mean() < 1e-5)):
      break
    Z_old = Z
  else:
    warnings.warn('ADMM failed to converge after %d iterations.' % max_iter)
  return C


def sparse_regularized_graph(X, positive=False, sparsity_param=None, kmax=None):
  '''Sparse Regularized Graph Construction, commonly known as an l1-graph.

  positive : bool, optional
    When True, computes the Sparse Probability Graph (SPG).
  sparsity_param : float, optional
    Controls sparsity cost in the LASSO optimization.
    When None, uses cross-validation to find sparsity parameters.
    This is very slow, but it gets good results.
  kmax : int, optional
    When None, allow all points to be edges. Otherwise, restrict to kNN set.

  l1-graph: "Semi-supervised Learning by Sparse Representation"
    Yan & Wang, SDM 2009
    http://epubs.siam.org/doi/pdf/10.1137/1.9781611972795.68

  SPG: "Nonnegative Sparse Coding for Discriminative Semi-supervised Learning"
    He et al., CVPR 2001
  '''
  clf, X = _l1_graph_setup(X, positive, sparsity_param)
  if kmax is None:
    W = _l1_graph_solve_full(clf, X)
  else:
    W = _l1_graph_solve_k(clf, X, kmax)
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
