import numpy as np
from scipy.spatial import Delaunay
from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import pairwise_distances, paired_distances
from graphs import Graph

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
  n = X.shape[0]
  a,b = np.triu_indices(n, k=1)
  D = pairwise_distances(X, metric=metric)
  # Naive algorithm, but it's generic to any D (doesn't depend on delaunay).
  pairs = []
  for pair in zip(a,b):
    d = D[pair]
    for i in xrange(n):
      if i in pair:
        continue
      if (D[pair,i] < d).all():
        break  # Point in lune, this is not an edge
    else:
      pairs.append(pair)
  return Graph.from_edge_pairs(pairs, num_vertices=X.shape[0], symmetric=True)
