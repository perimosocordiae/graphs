from __future__ import absolute_import

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances

try:
  from bottleneck import argpartsort
except ImportError:
  argpartsort = lambda arr, k: np.argpartition(arr, k-1)

from graphs import Graph

__all__ = ['neighbor_graph', 'nearest_neighbors']


def neighbor_graph(X, metric='euclidean', k=None, epsilon=None,
                   weighting='none', precomputed=False):
  '''Build a neighbor graph from pairwise distance information.

  X : two-dimensional array-like
      Shape must either be (num_pts, num_dims) or (num_pts, num_pts).
  k : int, maximum number of nearest neighbors
  epsilon : float, maximum distance to a neighbor
  metric : str, type of distance metric (see sklearn.metrics)
      When metric='precomputed', X is a symmetric distance matrix.
  weighting : str, one of {'binary', 'none'}
      When weighting='binary', all edge weights == 1.
  '''
  if k is None and epsilon is None:
    raise ValueError('Must provide `k` or `epsilon`.')
  if weighting not in ('binary', 'none'):
    raise ValueError('Invalid weighting param: %r' % weighting)

  # TODO: deprecate the precomputed kwarg
  precomputed = precomputed or (metric == 'precomputed')
  binary = weighting == 'binary'

  # Try the fast path, if possible.
  if not precomputed and epsilon is None:
    return _sparse_neighbor_graph(X, k, binary, metric)

  if precomputed:
    D = X
  else:
    D = pairwise_distances(X, metric=metric)
  return _slow_neighbor_graph(D, k, epsilon, binary)


def nearest_neighbors(query_pts, target_pts=None, metric='euclidean',
                      k=None, epsilon=None, return_dists=False,
                      precomputed=False):
  '''Find nearest neighbors of query points from a matrix of target points.

  Returns a list of indices of neighboring points, one list per query.
  If no target_pts are specified, distances are calculated within query_pts.
  When return_dists is True, returns two lists: (distances, indices)
  '''
  if k is None and epsilon is None:
    raise ValueError('Must provide `k` or `epsilon`.')

  # TODO: deprecate the precomputed kwarg
  precomputed = precomputed or (metric == 'precomputed')

  if precomputed and target_pts is not None:
    raise ValueError('`target_pts` cannot be used with precomputed distances')

  query_pts = np.array(query_pts)
  if len(query_pts.shape) == 1:
    query_pts = query_pts.reshape((1,-1))  # ensure that the query is a 1xD row

  if precomputed:
    dists = query_pts.copy()
  else:
    dists = pairwise_distances(query_pts, Y=target_pts, metric=metric)

  if epsilon is not None:
    if k is not None:
      # kNN filtering
      _, not_nn = _min_k_indices(dists, k, inv_ind=True)
      dists[np.arange(dists.shape[0]), not_nn.T] = np.inf
    # epsilon-ball
    is_close = dists <= epsilon
    if return_dists:
      nnis,nnds = [],[]
      for i,row in enumerate(is_close):
        nns = np.nonzero(row)[0]
        nnis.append(nns)
        nnds.append(dists[i,nns])
      return nnds, nnis
    return np.array([np.nonzero(row)[0] for row in is_close])

  # knn
  nns = _min_k_indices(dists,k)
  if return_dists:
    # index each row of dists by each row of nns
    row_inds = np.arange(len(nns))[:,np.newaxis]
    nn_dists = dists[row_inds, nns]
    return nn_dists, nns
  return nns


def _slow_neighbor_graph(dist, k, epsilon, binary):
  num_pts = dist.shape[0]

  if k is not None:
    k = min(k+1, num_pts)
    nn, not_nn = _min_k_indices(dist, k, inv_ind=True)
    I = np.arange(num_pts)

  if epsilon is not None:
    mask = dist <= epsilon
    if k is not None:
      mask[I, not_nn.T] = False
    if binary:
      np.fill_diagonal(mask, False)
      W = mask.astype(float)
    else:
      W = np.where(mask, dist, 0)
  else:
    inv_mask = np.eye(num_pts, dtype=bool)
    inv_mask[I, not_nn.T] = True
    if binary:
      W = 1.0 - inv_mask
    else:
      W = np.where(inv_mask, 0, dist)

  # W = scipy.sparse.csr_matrix(W)
  return Graph.from_adj_matrix(W)


def _min_k_indices(arr, k, inv_ind=False):
  psorted = argpartsort(arr, k)
  if inv_ind:
    return psorted[...,:k], psorted[...,k:]
  return psorted[...,:k]


def _sparse_neighbor_graph(X, k, binary=False, metric='l2'):
  '''Construct a sparse adj matrix from a matrix of points (one per row).
  Non-zeros are unweighted/binary distance values, depending on the binary arg.
  Doesn't include self-edges.'''
  knn = NearestNeighbors(n_neighbors=k, metric=metric).fit(X)
  mode = 'connectivity' if binary else 'distance'
  try:
    adj = knn.kneighbors_graph(None, mode=mode)
  except IndexError:
    # XXX: we must be running an old (<0.16) version of sklearn
    #  We have to hack around an old bug:
    if binary:
      adj = knn.kneighbors_graph(X, k+1, mode=mode)
      adj.setdiag(0)
    else:
      adj = knn.kneighbors_graph(X, k, mode=mode)
  return Graph.from_adj_matrix(adj)
