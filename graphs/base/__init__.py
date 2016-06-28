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
  if symmetric:
    G = SymmEdgePairGraph(pairs, num_vertices=num_vertices)
  else:
    G = EdgePairGraph(pairs, num_vertices=num_vertices)
  if weights is None:
    return G
  # Convert to sparse adj graph with provided edge weights
  s = G.matrix(csr=True, csc=True, coo=True).astype(float)
  # shenanigans to assign edge weights in the right order
  flat_idx = np.ravel_multi_index(s.nonzero(), s.shape)
  r, c = np.transpose(pairs)
  w_idx = np.ravel_multi_index((r,c), s.shape)
  s.data[np.searchsorted(flat_idx, w_idx)] = weights
  if symmetric:
    w_idx = np.ravel_multi_index((c,r), s.shape)
    s.data[np.searchsorted(flat_idx, w_idx)] = weights
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
