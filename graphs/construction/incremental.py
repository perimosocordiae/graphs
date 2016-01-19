from __future__ import absolute_import

import numpy as np
from sklearn.metrics import pairwise_distances

from graphs import Graph

__all__ = ['incremental_neighbor_graph']


def incremental_neighbor_graph(X, precomputed=False, k=None, epsilon=None,
                               weighting='none'):
  '''See neighbor_graph.'''
  assert ((k is not None) or (epsilon is not None)
          ), "Must provide `k` or `epsilon`"
  assert (_issequence(k) ^ _issequence(epsilon)
          ), "Exactly one of `k` or `epsilon` must be a sequence."
  assert weighting in ('binary','none'), "Invalid weighting param: " + weighting
  is_weighted = weighting == 'none'

  if precomputed:
    D = X
  else:
    D = pairwise_distances(X, metric='euclidean')
  # pre-sort for efficiency
  order = np.argsort(D)[:,1:]

  if k is None:
    k = D.shape[0]

  # generate the sequence of graphs
  # TODO: convert the core of these loops to Cython for speed
  W = np.zeros_like(D)
  I = np.arange(D.shape[0])
  if _issequence(k):
    # varied k, fixed epsilon
    if epsilon is not None:
      D[D > epsilon] = 0
    old_k = 0
    for new_k in k:
      idx = order[:, old_k:new_k]
      dist = D[I, idx.T]
      W[I, idx.T] = dist if is_weighted else 1
      yield Graph.from_adj_matrix(W)
      old_k = new_k
  else:
    # varied epsilon, fixed k
    idx = order[:,:k]
    dist = D[I, idx.T].T
    old_i = np.zeros(D.shape[0], dtype=int)
    for eps in epsilon:
      for i, row in enumerate(dist):
        oi = old_i[i]
        ni = oi + np.searchsorted(row[oi:], eps)
        rr = row[oi:ni]
        W[i, idx[i,oi:ni]] = rr if is_weighted else 1
        old_i[i] = ni
      yield Graph.from_adj_matrix(W)


def _issequence(x):
  # Note: isinstance(x, collections.Sequence) fails for numpy arrays
  return hasattr(x, '__len__')
