import numpy as np
import scipy.sparse as ss
import warnings

from base import Graph


class EdgePairGraph(Graph):
  def __init__(self, pairs, num_vertices=None):
    self._pairs = np.atleast_2d(pairs)
    # Handle empty-input case
    if self._pairs.size == 0:
      self._pairs.shape = (0, 2)
      self._pairs.dtype = int
      self._num_vertices = num_vertices if num_vertices is not None else 0
      return
    # Validate shape and dtype
    assert self._pairs.shape[1] == 2
    if not np.can_cast(self._pairs, int, casting='same_kind'):
      self._pairs = self._pairs.astype(int)
    # Set self._num_vertices
    if num_vertices is not None:
      self._num_vertices = num_vertices
    else:
      self._num_vertices = self._pairs.max() + 1

  def pairs(self, copy=False):
    if copy:
      return self._pairs.copy()
    return self._pairs

  def matrix(self, copy=False, **kwargs):
    n = self._num_vertices
    row,col = self.pairs().T
    data = np.ones(len(row), dtype=int)
    M = ss.coo_matrix((data, (row,col)), shape=(n,n))
    if not kwargs:
      return M
    if 'csr' in kwargs:
      return M.tocsr()
    if 'dense' in kwargs:
      return M.toarray()
    raise NotImplementedError('Unknown matrix type(s): %s' % kwargs.keys())

  def copy(self):
    return EdgePairGraph(self._pairs.copy(), num_vertices=self._num_vertices)

  def num_edges(self):
    return len(self._pairs)

  def num_vertices(self):
    return self._num_vertices

  def add_edges(self, from_idx, to_idx,
                weight=None, symmetric=False, copy=False):
    '''Adds all from->to edges.
    If symmetric=True, also adds to->from edges as well.'''
    if weight is not None:
      warnings.warn('Cannot supply weights for unweighted graph; '
                    'ignoring weight argument')
    to_add = np.column_stack((from_idx, to_idx))
    if symmetric:
      to_add = np.vstack((to_add, np.column_stack((to_idx, from_idx))))
    flattener = (self._num_vertices, 1)
    flat_inds = self._pairs.dot(flattener)
    flat_add = to_add.dot(flattener)
    if symmetric:
      _, idx = np.unique(flat_add, return_index=True)
      mask = np.zeros_like(flat_add, dtype=bool)
      mask[idx] = True
      flat_add = flat_add[mask]
    to_add = to_add[np.in1d(flat_add, flat_inds, invert=True)]
    res = self.copy() if copy else self
    if len(to_add) > 0:
      res._pairs = np.vstack((self._pairs, to_add))
    return res

  def reweight(self, weight, edges=None, copy=False):
    warnings.warn('Cannot supply weights for unweighted graph; '
                  'ignoring call to reweight')
    return self

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

  def pairs(self, copy=False):
    return np.vstack((self._pairs[self._offdiag_mask], self._pairs[:,::-1]))

  def num_edges(self):
    num_offdiag_edges = np.count_nonzero(self._offdiag_mask)
    return len(self._pairs) + num_offdiag_edges

  def copy(self):
    return SymmEdgePairGraph(self._pairs.copy(),
                             num_vertices=self._num_vertices,
                             ensure_format=False)

  def symmetrize(self, method=None, copy=False):
    if not copy:
      return self
    return SymmEdgePairGraph(self._pairs, num_vertices=self._num_vertices,
                             ensure_format=False)
