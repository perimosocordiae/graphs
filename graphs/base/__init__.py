from __future__ import absolute_import

import numpy as np
import scipy.sparse as ss

from .adj import SparseAdjacencyMatrixGraph, DenseAdjacencyMatrixGraph
from .base import Graph
from .pairs import EdgePairGraph, SymmEdgePairGraph

__all__ = ['Graph']


def from_edge_pairs(pairs, num_vertices=None, symmetric=False, weights=None):
  '''Constructor for Graph objects based on edges given as pairs of vertices.
  pairs : integer array-like with shape (num_edges, 2)
  '''
  if not symmetric:
    if weights is None:
      return EdgePairGraph(pairs, num_vertices=num_vertices)
    row, col = np.asarray(pairs).T
    row, weights = np.broadcast_arrays(row, weights)
    shape = None if num_vertices is None else (num_vertices, num_vertices)
    adj = ss.coo_matrix((weights, (row, col)), shape=shape)
    return SparseAdjacencyMatrixGraph(adj)
  # symmetric case
  G = SymmEdgePairGraph(pairs, num_vertices=num_vertices)
  if weights is None:
    return G
  # Convert to sparse adj graph with provided edge weights
  s = G.matrix('coo').astype(float)
  # shenanigans to assign edge weights in the right order
  flat_idx = np.ravel_multi_index(s.nonzero(), s.shape)
  r, c = np.transpose(pairs)
  rc_idx = np.ravel_multi_index((r,c), s.shape)
  cr_idx = np.ravel_multi_index((c,r), s.shape)
  order = np.argsort(flat_idx)
  flat_idx = flat_idx[order]
  s.data[order[np.searchsorted(flat_idx, rc_idx)]] = weights
  s.data[order[np.searchsorted(flat_idx, cr_idx)]] = weights
  return SparseAdjacencyMatrixGraph(s)


def from_adj_matrix(adj):
  '''Constructor for Graph objects based on a given adjacency matrix.
  adj : scipy.sparse matrix or array-like, shape (num_vertices, num_vertices)
  '''
  if ss.issparse(adj):
    return SparseAdjacencyMatrixGraph(adj)
  return DenseAdjacencyMatrixGraph(adj)

# Add static methods to the Graph class.
Graph.from_edge_pairs = staticmethod(from_edge_pairs)
Graph.from_adj_matrix = staticmethod(from_adj_matrix)
