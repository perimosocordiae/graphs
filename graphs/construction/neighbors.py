import numpy as np
import scipy.sparse
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances

try:
  from bottleneck import argpartsort
except ImportError:
  argpartsort = lambda arr, k: np.argpartition(arr, k-1)

from graphs import Graph


def neighbor_graph(X, precomputed=False, k=None, epsilon=None, symmetrize=True,
                   weighting='binary'):
  '''Construct an adj matrix from a matrix of points (one per row).
  When `precomputed` is True, X is a distance matrix.
  `weighting` param can be one of {binary, none}.'''
  assert ((k is not None) or (epsilon is not None)
          ), "Must provide `k` or `epsilon`"
  assert weighting in ('binary','none'), "Invalid weighting param: "+weighting

  # Try the fast path, if possible.
  if not precomputed and epsilon is None:
    is_binary = weighting == 'binary'
    G = _sparse_neighbor_graph(X, k, is_binary)
    if symmetrize:
      if is_binary:
        G.symmetrize(method='max')
      else:
        G.symmetrize(method='avg')
    return G
  # Dense/slow path.
  return _slow_neighbor_graph(X, precomputed, k, epsilon, symmetrize, weighting)


def _slow_neighbor_graph(X, precomputed, k, epsilon, symmetrize, weighting):
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

  if symmetrize and k is not None:
    dist = (dist + dist.T) / 2

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
  if binary:
    k += 1
  knn = NearestNeighbors(n_neighbors=k).fit(X)
  if binary:
    adj = knn.kneighbors_graph(X)
    adj.setdiag(0)
  else:
    adj = knn.kneighbors_graph(X, mode='distance')
  return Graph.from_adj_matrix(adj)
