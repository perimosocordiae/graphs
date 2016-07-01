from __future__ import absolute_import

import numpy as np
import scipy.sparse as ss
import warnings

from .base import Graph


class EdgePairGraph(Graph):
  def __init__(self, pairs, num_vertices=None):
    self._pairs = np.atleast_2d(pairs)
    # Handle empty-input case
    if self._pairs.size == 0:
      self._pairs.shape = (0, 2)
      self._pairs = self._pairs.astype(np.intp, copy=False)
      self._num_vertices = num_vertices if num_vertices is not None else 0
      return
    # Validate shape and dtype
    assert self._pairs.shape[1] == 2
    if not np.can_cast(self._pairs, np.intp, casting='same_kind'):
      self._pairs = self._pairs.astype(np.intp)
    # Set self._num_vertices
    if num_vertices is not None:
      self._num_vertices = num_vertices
    else:
      self._num_vertices = self._pairs.max() + 1

  def pairs(self, copy=False, directed=True):
    if not directed:
      canonical = np.sort(self._pairs, axis=1)
      n = self._num_vertices
      _, uniq_idx = np.unique(np.ravel_multi_index(canonical.T, (n,n)),
                              return_index=True)
      return canonical[uniq_idx]
    if copy:
      return self._pairs.copy()
    return self._pairs

  def matrix(self, copy=False, **kwargs):
    n = self._num_vertices
    row,col = self.pairs().T
    data = np.ones(len(row), dtype=np.intp)
    M = ss.coo_matrix((data, (row,col)), shape=(n,n))
    if not kwargs:
      return M
    if 'csr' in kwargs:
      return M.tocsr()
    if 'dense' in kwargs:
      return M.toarray()
    raise NotImplementedError('Unknown matrix type(s): %s' % (
                              tuple(kwargs.keys()),))

  def copy(self):
    return EdgePairGraph(self._pairs.copy(), num_vertices=self._num_vertices)

  def num_edges(self):
    return len(self._pairs)

  def num_vertices(self):
    return self._num_vertices

  def add_edges(self, from_idx, to_idx,
                weight=None, symmetric=False, copy=False):
    if weight is not None:
      warnings.warn('Cannot supply weights for unweighted graph; '
                    'ignoring weight argument')
    to_add = np.column_stack((from_idx, to_idx))
    if symmetric:
      # add reversed edges, excluding diagonals
      diag_mask = np.not_equal(*to_add.T)
      rev = to_add[diag_mask,::-1]
      to_add = np.vstack((to_add, rev))
    # select only those edges that are not already present
    flattener = (self._num_vertices, 1)
    flat_inds = self._pairs.dot(flattener)
    flat_add = to_add.dot(flattener)
    to_add = to_add[np.in1d(flat_add, flat_inds, invert=True)]
    # add the new edges
    res = self.copy() if copy else self
    if len(to_add) > 0:
      res._pairs = np.vstack((self._pairs, to_add))
    return res

  def remove_edges(self, from_idx, to_idx, symmetric=False, copy=False):
    from_idx, to_idx = np.atleast_1d(from_idx, to_idx)
    flat_inds = self._pairs.dot((self._num_vertices, 1))
    to_remove = from_idx * self._num_vertices + to_idx
    if symmetric:
      to_remove = np.concatenate((to_remove,
                                  to_idx * self._num_vertices + from_idx))
    mask = np.in1d(flat_inds, to_remove, invert=True)
    res = self.copy() if copy else self
    res._pairs = res._pairs[mask]
    return res

  def symmetrize(self, method=None, copy=False):
    '''Symmetrizes (ignores method). Returns a copy if copy=True.'''
    if copy:
      return SymmEdgePairGraph(self._pairs.copy(),
                               num_vertices=self._num_vertices)
    shape = (self._num_vertices, self._num_vertices)
    flat_inds = np.union1d(np.ravel_multi_index(self._pairs.T, shape),
                           np.ravel_multi_index(self._pairs.T[::-1], shape))
    self._pairs = np.transpose(np.unravel_index(flat_inds, shape))
    return self

  def subgraph(self, mask):
    nv = self.num_vertices()
    idx = np.arange(nv)[mask]
    idx_map = np.full(nv, -1)
    idx_map[idx] = np.arange(len(idx))
    pairs = idx_map[self._pairs]
    pairs = pairs[(pairs >= 0).all(axis=1)]
    return EdgePairGraph(pairs, num_vertices=len(idx))

  pairs.__doc__ = Graph.pairs.__doc__
  matrix.__doc__ = Graph.matrix.__doc__
  add_edges.__doc__ = Graph.add_edges.__doc__
  remove_edges.__doc__ = Graph.remove_edges.__doc__
  subgraph.__doc__ = Graph.subgraph.__doc__


class SymmEdgePairGraph(EdgePairGraph):
  def __init__(self, pairs, num_vertices=None, ensure_format=True):
    EdgePairGraph.__init__(self, pairs, num_vertices=num_vertices)
    if ensure_format:
      # push all edges to upper triangle
      self._pairs.sort()
      # remove any duplicates
      shape = (self._num_vertices, self._num_vertices)
      _, idx = np.unique(np.ravel_multi_index(self._pairs.T, shape),
                         return_index=True)
      self._pairs = self._pairs[idx]
    self._offdiag_mask = np.not_equal(*self._pairs.T)

  def pairs(self, copy=False, directed=True):
    if directed:
      return np.vstack((self._pairs[self._offdiag_mask], self._pairs[:,::-1]))
    if copy:
      return self._pairs.copy()
    return self._pairs

  def num_edges(self):
    num_offdiag_edges = np.count_nonzero(self._offdiag_mask)
    return len(self._pairs) + num_offdiag_edges

  def copy(self):
    return SymmEdgePairGraph(self._pairs.copy(),
                             num_vertices=self._num_vertices,
                             ensure_format=False)

  def remove_edges(self, from_idx, to_idx, symmetric=False, copy=False):
    '''Removes all from->to and to->from edges.
    Note: the symmetric kwarg is unused.'''
    flat_inds = self._pairs.dot((self._num_vertices, 1))
    # convert to sorted order and flatten
    to_remove = (np.minimum(from_idx, to_idx) * self._num_vertices
                 + np.maximum(from_idx, to_idx))
    mask = np.in1d(flat_inds, to_remove, invert=True)
    res = self.copy() if copy else self
    res._pairs = res._pairs[mask]
    res._offdiag_mask = res._offdiag_mask[mask]
    return res

  def symmetrize(self, method=None, copy=False):
    '''Alias for copy()'''
    if not copy:
      return self
    return SymmEdgePairGraph(self._pairs, num_vertices=self._num_vertices,
                             ensure_format=False)

  def subgraph(self, mask):
    g = EdgePairGraph.subgraph(mask)
    return SymmEdgePairGraph(g._pairs, num_vertices=g._num_vertices,
                             ensure_format=False)

  pairs.__doc__ = Graph.pairs.__doc__
  subgraph.__doc__ = Graph.subgraph.__doc__
