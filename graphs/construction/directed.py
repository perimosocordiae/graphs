from __future__ import absolute_import, print_function

import numpy as np
from sklearn.metrics.pairwise import paired_distances
from graphs import Graph
from .neighbors import neighbor_graph

__all__ = ['directed_graph']


def directed_graph(trajectories, k=5, verbose=False, pruning_thresh=0,
                   return_coords=False):
  '''Directed graph construction alg. from Johns & Mahadevan, ICML '07.
  trajectories: list of NxD arrays of ordered states
  '''
  X = np.vstack(trajectories)
  G = neighbor_graph(X, k=k)
  if pruning_thresh > 0:
    traj_len = map(len, trajectories)
    G = _prune_edges(G, X, traj_len, pruning_thresh, verbose=verbose)
  if return_coords:
    return G, X
  return G


def _prune_edges(G, X, traj_lengths, pruning_thresh=0.1, verbose=False):
  '''Prune edges in graph G via cosine distance with trajectory edges.'''
  W = G.matrix(dense=True).copy()
  degree = G.degree(kind='out', weighted=False)
  i = 0
  num_bad = 0
  for n in traj_lengths:
    s, t = np.nonzero(W[i:i+n-1])
    graph_edges = X[t] - X[s+i]
    traj_edges = np.diff(X[i:i+n], axis=0)
    traj_edges = np.repeat(traj_edges, degree[i:i+n-1], axis=0)
    theta = paired_distances(graph_edges, traj_edges, 'cosine')
    bad_edges = theta > pruning_thresh
    s, t = s[bad_edges], t[bad_edges]
    if verbose:  # pragma: no cover
      num_bad += np.count_nonzero(W[s,t])
    W[s,t] = 0
    i += n
  if verbose:  # pragma: no cover
    print('removed %d bad edges' % num_bad)
  return Graph.from_adj_matrix(W)
