'''Sparse symmetric matrix reordering to reduce bandwidth/diagonalness.
Methods:
 - cuthill_mckee
 - node_centroid_hill_climbing
 - laplacian_reordering
References:
 - ftp://ftp.numerical.rl.ac.uk/pub/talks/jas.ala06.24VII06.pdf
 - http://www.jstor.org/stable/2156090 (profile defn, NYI RCM improvements)
 - https://www.cs.purdue.edu/homes/apothen/env3.pdf (laplacian, NYI sloan alg)
'''
from __future__ import absolute_import, print_function
from collections import deque
import numpy as np
import scipy.sparse.csgraph as ssc
from graphs import Graph
from .mini_six import range

__all__ = [
    'permute_graph', 'cuthill_mckee', 'node_centroid_hill_climbing',
    'laplacian_reordering'
]


def permute_graph(G, order):
  '''Reorder the graph's vertices, returning a copy of the input graph.
  order : integer array-like, some permutation of range(G.num_vertices()).
  '''
  adj = G.matrix(dense=True)
  adj = adj[np.ix_(order, order)]
  return Graph.from_adj_matrix(adj)


def _cuthill_mckee(G):
  n = G.num_vertices()
  queue = deque([])
  result = []
  degree = G.degree()
  remaining = dict(enumerate(degree))
  adj = G.matrix(dense=True, csr=True)
  while len(result) != n:
    queue.append(min(remaining, key=remaining.get))
    while queue:
      p = queue.popleft()
      if p not in remaining:
        continue
      result.append(p)
      del remaining[p]
      nbrs = [c for c in np.where(adj[p])[0] if c in remaining]
      queue.extend(sorted(nbrs, key=remaining.get))
  return permute_graph(G, np.array(result))


if hasattr(ssc, 'reverse_cuthill_mckee'):  # pragma: no cover
  def cuthill_mckee(G):
    sG = G.matrix(csr=True)
    order = ssc.reverse_cuthill_mckee(sG, symmetric_mode=True)
    return permute_graph(G, order)
else:  # pragma: no cover
  cuthill_mckee = _cuthill_mckee

cuthill_mckee.__doc__ = 'Reorder vertices using the Cuthill-McKee algorithm.'


def laplacian_reordering(G):
  '''Reorder vertices using the eigenvector of the graph Laplacian corresponding
  to the first positive eigenvalue.'''
  L = G.laplacian()
  vals, vecs = np.linalg.eigh(L)
  min_positive_idx = np.argmax(vals == vals[vals>0].min())
  vec = vecs[:, min_positive_idx]
  return permute_graph(G, np.argsort(vec))


def node_centroid_hill_climbing(G, relax=1, num_centerings=20, verbose=False):
  '''Iterative reordering method based on alternating rounds of node-centering
  and hill-climbing search.'''
  # Initialize order with BFS from a random start node.
  order = _breadth_first_order(G)
  for it in range(num_centerings):
    B = permute_graph(G, order).bandwidth()
    nc_order = _node_center(G, order, relax=relax)
    nc_B = permute_graph(G, nc_order).bandwidth()
    if nc_B < B:
      if verbose:  # pragma: no cover
        print('post-center', B, nc_B)
      order = nc_order
    order = _hill_climbing(G, order, verbose=verbose)
  return permute_graph(G, order)


def _breadth_first_order(G):
  inds = np.arange(G.num_vertices())
  adj = G.matrix(dense=True, csr=True)
  total_order = []
  while len(inds) > 0:
    order = ssc.breadth_first_order(adj, np.random.choice(inds),
                                    return_predecessors=False)
    inds = np.setdiff1d(inds, order, assume_unique=True)
    total_order = np.append(total_order, order)
  return total_order.astype(int)


def _critical_vertices(G, order, relax=1, bw=None):
  go = permute_graph(G, order)
  if bw is None:
    bw = go.bandwidth()
  adj = go.matrix(dense=True)
  if relax == 1:
    for i in np.where(np.diag(adj, -bw))[0]:
      yield bw + i, i
  else:
    crit = relax * bw
    for u, v in np.transpose(np.where(np.tril(adj, -np.floor(crit)))):
      if np.abs(u-v) >= crit:
        yield u, v


def _node_center(G, order, relax=0.99):
  weights = order.copy().astype(float)
  counts = np.ones_like(order)
  inv_order = np.argsort(order)
  for i, j in _critical_vertices(G, order, relax):
    u = inv_order[i]
    v = inv_order[j]
    weights[u] += j  # order[v]
    counts[u] += 1
    weights[v] += i  # order[u]
    counts[v] += 1
  weights /= counts
  return np.argsort(weights)


def _hill_climbing(G, order, verbose=False):
  B = permute_graph(G, order).bandwidth()
  while True:
    inv_order = np.argsort(order)
    for i, j in _critical_vertices(G, order, bw=B):
      u = inv_order[i]
      v = inv_order[j]
      for w,k in enumerate(order):
        if not (k < i or k > j):
          continue
        new_order = order.copy()
        if k < i:
          new_order[[u,w]] = new_order[[w,u]]
        elif k > j:
          new_order[[v,w]] = new_order[[w,v]]

        new_B = permute_graph(G, new_order).bandwidth()
        if new_B < B:
          order = new_order
          if verbose:  # pragma: no cover
            print('improved B', B, new_B)
          B = new_B
          break
        elif new_B == B:
          nc = sum(1 for _ in _critical_vertices(G, order, bw=B))
          new_nc = sum(1 for _ in _critical_vertices(G, new_order, bw=B))
          if new_nc < nc:
            order = new_order
            if verbose:  # pragma: no cover
              print('improved nc', nc, new_nc)
            break
      else:
        continue
      break
    else:
      break
  return order
