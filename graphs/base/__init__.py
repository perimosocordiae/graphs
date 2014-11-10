import numpy as np
import scipy.sparse as ss

from adj import SparseAdjacencyMatrixGraph, DenseAdjacencyMatrixGraph
from base import Graph
from pairs import EdgePairGraph, SymmEdgePairGraph

__all__ = ['Graph']


def from_edge_pairs(pairs, num_vertices=None, symmetric=False, weights=None):
  if symmetric:
    assert weights is None, 'symmetric+weighted from_edge_pairs is NYI'
    return SymmEdgePairGraph(pairs, num_vertices=num_vertices)
  G = EdgePairGraph(pairs, num_vertices=num_vertices)
  if weights is None:
    return G
  # Convert to sparse adj graph with provided edge weights
  s = G.matrix(csr=True, csc=True, coo=True).astype(float)
  # shenanigans to assign edge weights in the right order
  nv = G.num_vertices()
  order = np.argsort((pairs * np.array([nv, 1])).sum(axis=1))
  s.data[:] = weights[order]
  return SparseAdjacencyMatrixGraph(s)


def from_adj_matrix(adj, weighted=True):
  assert weighted, 'Unweighted adj-matrix graphs are NYI'
  if ss.issparse(adj):
    return SparseAdjacencyMatrixGraph(adj)
  return DenseAdjacencyMatrixGraph(adj)

# Add static methods to the Graph class.
Graph.from_edge_pairs = staticmethod(from_edge_pairs)
Graph.from_adj_matrix = staticmethod(from_adj_matrix)
