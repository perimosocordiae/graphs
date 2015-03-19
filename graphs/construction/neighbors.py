import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances

try:
  from bottleneck import argpartsort
except ImportError:
  argpartsort = lambda arr, k: np.argpartition(arr, k-1)

from graphs import Graph

__all__ = ['neighbor_graph', 'nearest_neighbors']


def neighbor_graph(X, precomputed=False, k=None, epsilon=None,
                   weighting='none'):
  '''Construct an adj matrix from a matrix of points (one per row).
  When `precomputed` is True, X is a distance matrix.
  `weighting` param can be one of {binary, none}.'''
  assert ((k is not None) or (epsilon is not None)
          ), "Must provide `k` or `epsilon`"
  assert weighting in ('binary','none'), "Invalid weighting param: " + weighting

  # Try the fast path, if possible.
  if not precomputed and epsilon is None:
    G = _sparse_neighbor_graph(X, k, weighting == 'binary')
  else:
    G = _slow_neighbor_graph(X, precomputed, k, epsilon, weighting)
  return G


def nearest_neighbors(query_pts, target_pts=None, precomputed=False,
                      k=None, epsilon=None, return_dists=False):
  '''Find nearest neighbors of query points from a matrix of target points.
    Returns a list of indices of neighboring points, one list per query.
    If no target_pts are specified, distances are calculated within query_pts.
    When return_dists is True, returns two lists: (indices,distances)'''
  # sanity checks
  assert ((k is not None) or (epsilon is not None)
          ), "Must provide `k` or `epsilon`"
  query_pts = np.array(query_pts)
  if len(query_pts.shape) == 1:
    query_pts = query_pts.reshape((1,-1))  # ensure that the query is a 1xD row
  if precomputed:
    dists = query_pts.copy()
    assert (target_pts is None
            ), 'target_pts should not be provided with precomputed=True'
  elif target_pts is None:
    dists = pairwise_distances(query_pts, metric='euclidean')
  else:
    dists = pairwise_distances(query_pts, target_pts, metric='euclidean')
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


def _slow_neighbor_graph(X, precomputed, k, epsilon, weighting):
  num_pts = X.shape[0]
  if precomputed:
    dist = X.copy()
  else:
    dist = pairwise_distances(X, metric='euclidean')

  if k is not None:
    k = min(k+1, num_pts)
    nn, not_nn = _min_k_indices(dist, k, inv_ind=True)

  if epsilon is not None:
    if k is not None:
      dist[np.arange(dist.shape[0]), not_nn.T] = np.inf
    dist[dist>epsilon] = 0  # zero out neighbors too far away
  else:
    for i in xrange(num_pts):
      dist[i,not_nn[i]] = 0  # zero out neighbors too far away

  if weighting is 'binary':
    dist = dist.astype(bool).astype(float)
  # dist = scipy.sparse.csr_matrix(dist)
  return Graph.from_adj_matrix(dist)


def _min_k_indices(arr, k, inv_ind=False):
  psorted = argpartsort(arr, k)
  if inv_ind:
    return psorted[...,:k], psorted[...,k:]
  return psorted[...,:k]


def _sparse_neighbor_graph(X, k, binary=False):
  '''Construct a sparse adj matrix from a matrix of points (one per row).
  Non-zeros are unweighted/binary distance values, depending on the binary arg.
  Doesn't include self-edges.'''
  knn = NearestNeighbors(n_neighbors=k+1).fit(X)
  if binary:
    adj = knn.kneighbors_graph(X)
    adj.setdiag(0)
  else:
    adj = knn.kneighbors_graph(X, mode='distance')
  return Graph.from_adj_matrix(adj)
