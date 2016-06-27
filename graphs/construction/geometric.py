from __future__ import absolute_import, print_function

import numpy as np
from scipy.spatial import Delaunay
from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import pairwise_distances, paired_distances
from graphs import Graph
from ..mini_six import range

__all__ = ['delaunay_graph', 'gabriel_graph', 'relative_neighborhood_graph']


def delaunay_graph(X):
  tri = Delaunay(X)
  n = X.shape[0]
  rows = np.empty(2*np.product(tri.simplices.shape), dtype=np.intp)
  cols = np.empty_like(rows)
  data = np.ones_like(rows, dtype=bool)
  i = 0
  d = tri.simplices.shape[1]
  for corners in tri.simplices:
    rows[i:i+d] = corners
    cols[i:i+d-1] = corners[1:]
    i += d
    cols[i-1] = corners[0]
  rows[i:] = cols[:i]
  cols[i:] = rows[:i]
  adj = coo_matrix((data, (rows, cols)), shape=(n,n))
  return Graph.from_adj_matrix(adj)


def gabriel_graph(X, metric='euclidean'):
  a,b = np.triu_indices(X.shape[0], k=1)
  midpoints = (X[a] + X[b]) / 2
  Dmid = pairwise_distances(midpoints, X, metric=metric).min(axis=1)
  Dedge = paired_distances(X[a], X[b], metric=metric)
  mask = (Dedge - Dmid * 2) < 1e-10
  pairs = np.transpose((a[mask],b[mask]))
  return Graph.from_edge_pairs(pairs, num_vertices=X.shape[0], symmetric=True)


def relative_neighborhood_graph(X, metric='euclidean'):
  D = pairwise_distances(X, metric=metric)
  pairs = find_relative_neighbors(D)
  return Graph.from_edge_pairs(pairs, num_vertices=D.shape[0], symmetric=True)


def _find_relative_neighbors(D):
  # Naive algorithm, but it's generic to any D (doesn't depend on delaunay).
  n = D.shape[0]
  pairs = []
  for r in range(n-1):
    for c in range(r+1, n):
      d = D[r,c]
      for i in range(n):
        if i == r or i == c:
          continue
        if D[r,i] < d and D[c,i] < d:
          break  # Point in lune, this is not an edge
      else:
        pairs.append((r,c))
  return pairs


try:
  import pyximport
  pyximport.install(setup_args={'include_dirs': np.get_include()})
  from ._fast_paths import find_relative_neighbors
except ImportError:
  find_relative_neighbors = _find_relative_neighbors
