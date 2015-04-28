import numpy as np
from scipy.sparse import coo_matrix
from graphs import Graph

__all__ = ['random_graph']


def random_graph(out_degree):
  '''Random graph generator. Does not generate self-edges.
  out_degree : array-like of ints, controlling the out degree of each vertex.
  '''
  n = len(out_degree)
  out_degree = np.asarray(out_degree, dtype=int)
  if (out_degree >= n).any():
    raise ValueError('Cannot have degree >= num_vertices')
  row = np.repeat(np.arange(n), out_degree)
  weights = np.ones_like(row, dtype=float)
  # Generate random edges from 0 to n-2, then shift by one to avoid self-edges.
  col = np.concatenate([np.random.choice(n-1, d, replace=False)
                        for d in out_degree])
  col[col >= row] += 1
  adj = coo_matrix((weights, (row, col)), shape=(n, n))
  return Graph.from_adj_matrix(adj)
