import numpy as np
import scipy.sparse as ss
import warnings

from base import Graph, _symmetrize


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

  def num_edges(self):
    return len(self._pairs)

  def num_vertices(self):
    return self._num_vertices

  def add_self_edges(self, weight=None):
    '''Adds all i->i edges, in-place.'''
    if weight is not None:
      warnings.warn('Cannot supply weights for unweighted graph; '
                    'ignoring weight argument')
    row,col = self._pairs.T
    diag_inds = row[np.equal(row,col)]
    to_add = np.arange(self._num_vertices)
    to_add = np.setdiff1d(to_add, diag_inds, assume_unique=True)
    if len(to_add) > 0:
      self._pairs = np.vstack((self._pairs, np.tile(to_add, (2,1)).T))
    return self

  def add_edges(self, from_idx, to_idx, weight=None, symmetric=False):
    '''Adds all from->to edges, in-place.'''
    if weight is not None:
      warnings.warn('Cannot supply weights for unweighted graph; '
                    'ignoring weight argument')
    to_add = np.column_stack((from_idx, to_idx))
    flattener = (self._num_vertices, 1)
    flat_inds = self._pairs.dot(flattener)
    flat_add = to_add.dot(flattener)
    to_add = to_add[~np.in1d(flat_add, flat_inds)]
    if len(to_add) > 0:
      self._pairs = np.vstack((self._pairs, to_add))
    return self

  def symmetrize(self, overwrite=True, method='sum'):
    '''Symmetrizes with the given method. {sum,max,avg}
    Returns a copy if overwrite=False.'''
    # TODO: be smarter about this and return a SymmEdgePairGraph
    S = _symmetrize(self.matrix(dense=True, csr=True), method)
    P = np.transpose(np.nonzero(S))
    if overwrite:
      self._pairs = P
      return self
    return EdgePairGraph(P, num_vertices=self._num_vertices)


class SymmEdgePairGraph(EdgePairGraph):
  def __init__(self, pairs, num_vertices=None):
    EdgePairGraph.__init__(self, pairs, num_vertices=num_vertices)
    self._pairs.sort()  # push all edges to upper triangle
    self._offdiag_mask = ~np.equal(*self._pairs.T)

  def pairs(self, copy=False):
    return np.vstack((self._pairs[self._offdiag_mask], self._pairs[:,::-1]))

  def num_edges(self):
    num_offdiag_edges = np.count_nonzero(self._offdiag_mask)
    return len(self._pairs) + num_offdiag_edges

  def symmetrize(self, overwrite=True, method='sum'):
    if overwrite:
      return self
    return SymmEdgePairGraph(self._pairs, num_vertices=self._num_vertices)
