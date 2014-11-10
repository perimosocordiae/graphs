import numpy as np
import scipy.sparse as ss

from base import Graph, _symmetrize


class AdjacencyMatrixGraph(Graph):

  def pairs(self, copy=False):
    return np.transpose(np.nonzero(self._adj))

  def num_vertices(self):
    return self._adj.shape[0]

  def is_weighted(self):
    return True


class DenseAdjacencyMatrixGraph(AdjacencyMatrixGraph):
  def __init__(self, adj):
    self._adj = np.atleast_2d(adj)
    assert self._adj.shape[0] == self._adj.shape[1]

  def matrix(self, copy=False, **kwargs):
    if not kwargs or 'dense' in kwargs:
      if copy:
        return self._adj.copy()
      return self._adj
    if 'csr' in kwargs:
      return ss.csr_matrix(self._adj)
    raise NotImplementedError('Unknown matrix type(s): %s' % kwargs.keys())

  def edge_weights(self, copy=False):
    ii,jj = self.pairs().T
    W = self._adj[ii,jj]
    if copy:
      return W.copy()
    return W

  def num_edges(self):
    return np.count_nonzero(self._adj)

  def add_self_edges(self, weight=1):
    '''Adds all i->i edges, in-place.'''
    # Do some dtype checking shenanigans.
    if not isinstance(weight, int):
      self._adj = self._adj.astype(float)
    np.fill_diagonal(self._adj, weight)
    return self

  def add_edges(self, from_idx, to_idx, weight=1, symmetric=False):
    '''Adds all from->to edges, in-place.'''
    # TODO
    return self

  def symmetrize(self, overwrite=True, method='sum'):
    '''Symmetrizes with the given method. {sum,max,avg}
    Returns a copy if overwrite=False.'''
    S = _symmetrize(self._adj, method)
    if overwrite:
      self._adj = S
      return self
    return DenseAdjacencyMatrixGraph(S)


class SparseAdjacencyMatrixGraph(AdjacencyMatrixGraph):
  def __init__(self, adj):
    assert ss.issparse(adj), 'SparseAdjacencyMatrixGraph input must be sparse'
    self._adj = adj
    assert self._adj.shape[0] == self._adj.shape[1]
    # Things go wrong if we have explicit zeros in the graph.
    if adj.format in ('csr', 'csc'):
      self._adj.eliminate_zeros()

  def matrix(self, copy=False, **kwargs):
    assert ss.issparse(self._adj), 'SparseAdjacencyMatrixGraph must be sparse'
    if not kwargs or self._adj.format in kwargs:
      if copy:
        return self._adj.copy()
      return self._adj
    for fmt in kwargs:
      if fmt != 'dense' and hasattr(self._adj, 'to'+fmt):
        return getattr(self._adj, 'to'+fmt)()
    if 'dense' in kwargs:
      return self._adj.toarray()
    raise NotImplementedError('Unknown matrix type(s): %s' % kwargs.keys())

  def edge_weights(self, copy=False):
    W = self._adj.data.ravel()  # assumes correct internal ordering
    # Also assumes no explicit zeros
    if copy:
      return W.copy()
    return W

  def num_edges(self):
    return self._adj.nnz

  def add_self_edges(self, weight=1):
    '''Adds all i->i edges, in-place.'''
    # Do some dtype checking shenanigans.
    if not isinstance(weight, int):
      self._adj = self._adj.astype(float)
    try:
      self._adj.setdiag(weight)
    except TypeError:
      # Older scipy doesn't support setdiag on everything.
      self._adj = self._adj.tocsr()
      self._adj.setdiag(weight)
    if weight == 0:
      # TODO: be smarter about avoiding writing explicit zeros
      # We changed the sparsity structure, possibly.
      assert hasattr(self._adj, 'eliminate_zeros'), 'Other formats NYI'
      self._adj.eliminate_zeros()
    return self

  def add_edges(self, from_idx, to_idx, weight=1, symmetric=False):
    '''Adds all from->to edges, in-place.'''
    # TODO
    return self

  def symmetrize(self, overwrite=True, method='sum'):
    '''Symmetrizes with the given method. {sum,max,avg}
    Returns a copy if overwrite=False.'''
    S = _symmetrize(self._adj.tocsr(), method)
    if overwrite:
      self._adj = S
      return self
    return SparseAdjacencyMatrixGraph(S)
