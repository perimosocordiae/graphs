from __future__ import absolute_import
import numpy as np
import scipy.sparse as ss

from .. import Graph


def chain_graph(num_vertices, wraparound=False, directed=False, weights=None):
  if wraparound:
    ii = np.arange(num_vertices)
    jj = ii + 1
    jj[-1] = 0
  else:
    ii = np.arange(num_vertices-1)
    jj = ii + 1
  pairs = np.column_stack((ii, jj))
  return Graph.from_edge_pairs(pairs, num_vertices=num_vertices,
                               symmetric=(not directed), weights=weights)


def lattice_graph(dims, wraparound=False):
  dims = [d for d in dims if d > 1]
  if len(dims) == 0:
    raise ValueError('Must supply at least one dimension >= 2')
  if len(dims) == 1:
    return chain_graph(dims[0], wraparound=wraparound)
  if len(dims) > 2:  # pragma: no cover
    raise NotImplementedError('NYI: len(dims) > 2')

  # 2d case
  m, n = dims
  num_vertices = m * n
  if wraparound:
    offsets = [-m*(n-1), -m, -m+1, -1, 1, m-1, m, m*(n-1)]
    data = np.ones((8, num_vertices), dtype=int)
    data[[2,5], :] = 0
    data[2, ::m] = 1
    data[3, m-1::m] = 0
    data[4, ::m] = 0
    data[5, m-1::m] = 1
    # handle edge cases where offsets are duplicated
    offsets, idx = np.unique(offsets, return_index=True)
    data = data[idx]
  else:
    offsets = [-m, -1, 1, m]
    data = np.ones((4, num_vertices), dtype=int)
    data[1, m-1::m] = 0
    data[2, 0::m] = 0
  adj = ss.dia_matrix((data, offsets), shape=(num_vertices, num_vertices))
  return Graph.from_adj_matrix(adj)
