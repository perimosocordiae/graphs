from __future__ import absolute_import, print_function

import numpy as np
from scipy.spatial import Delaunay
from sklearn.metrics.pairwise import (
    pairwise_distances, paired_distances, pairwise_distances_argmin_min)
from graphs import Graph
from ..mini_six import range

__all__ = [
    'delaunay_graph', 'urquhart_graph', 'gabriel_graph',
    'relative_neighborhood_graph'
]


def delaunay_graph(X, weighted=False):
  '''Delaunay triangulation graph.
  '''
  e1, e2 = _delaunay_edges(X)
  pairs = np.column_stack((e1, e2))
  w = paired_distances(X[e1], X[e2]) if weighted else None
  return Graph.from_edge_pairs(pairs, num_vertices=X.shape[0], symmetric=True,
                               weights=w)


def urquhart_graph(X, weighted=False):
  '''Urquhart graph: made from the 2 shortest edges of each Delaunay triangle.
  '''
  e1, e2 = _delaunay_edges(X)
  w = paired_distances(X[e1], X[e2])
  mask = np.ones_like(w, dtype=bool)
  bad_inds = w.reshape((-1, 3)).argmax(axis=1) + np.arange(0, len(e1), 3)
  mask[bad_inds] = False

  weights = w[mask] if weighted else None
  pairs = np.column_stack((e1[mask], e2[mask]))
  return Graph.from_edge_pairs(pairs, num_vertices=X.shape[0], symmetric=True,
                               weights=weights)


def gabriel_graph(X, metric='euclidean', weighted=False):
  n = X.shape[0]
  a, b = np.triu_indices(n, k=1)
  midpoints = (X[a] + X[b]) / 2
  _, Dmid = pairwise_distances_argmin_min(midpoints, X, metric=metric)
  Dedge = paired_distances(X[a], X[b], metric=metric)
  mask = (Dedge - Dmid * 2) < 1e-10
  pairs = np.column_stack((a[mask], b[mask]))
  w = Dedge[mask] if weighted else None
  return Graph.from_edge_pairs(pairs, num_vertices=n, symmetric=True, weights=w)


def relative_neighborhood_graph(X, metric='euclidean', weighted=False):
  D = pairwise_distances(X, metric=metric)
  n = D.shape[0]
  pairs = np.asarray(find_relative_neighbors(D))
  w = D[pairs[:,0],pairs[:,1]] if weighted else None
  return Graph.from_edge_pairs(pairs, num_vertices=n, symmetric=True, weights=w)


def _delaunay_edges(X):
  tri = Delaunay(X)
  e1 = tri.simplices.ravel()
  e2 = np.roll(tri.simplices, 1, axis=1).ravel()
  return e1, e2


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
  from ._fast_paths import find_relative_neighbors
except ImportError:
  try:
    import pyximport
    pyximport.install(setup_args={'include_dirs': np.get_include()})
    from ._fast_paths import find_relative_neighbors
  except ImportError:
    find_relative_neighbors = _find_relative_neighbors
