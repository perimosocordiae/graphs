import numpy as np
import scipy.sparse as ss

__all__ = ['Graph']


class Graph(object):

  def __init__(self, *args, **kwargs):
    raise NotImplementedError('Graph should not be instantiated directly')

  def pairs(self, copy=False):
    raise NotImplementedError()

  def matrix(self, copy=False, **kwargs):
    raise NotImplementedError()

  def edge_weights(self, copy=False):
    raise NotImplementedError()

  def num_edges(self):
    raise NotImplementedError()

  def num_vertices(self):
    raise NotImplementedError()

  def add_self_edges(self, weight=None):
    raise NotImplementedError()

  def symmetrize(self, overwrite=True, method='sum'):
    raise NotImplementedError()

  def is_weighted(self):
    return False

  def is_directed(self):
    return True

  def adj_list(self):
    '''Generates a sequence of lists of neighbor indices:
        an adjacency list representation.'''
    adj = self.matrix(dense=True, csr=True)
    for row in adj:
      yield row.nonzero()[-1]

  def degree(self, kind='out', unweighted=False):
    axis = 1 if kind == 'out' else 0
    adj = self.matrix(dense=True, csr=1-axis, csc=axis)
    if unweighted and self.is_weighted():
      # With recent numpy and a dense matrix, could do:
      # d = np.count_nonzero(adj, axis=axis)
      d = (adj!=0).sum(axis=axis)
    else:
      d = adj.sum(axis=axis)
    return np.asarray(d).ravel()

  @staticmethod
  def from_edge_pairs(pairs, num_vertices=None):
    return EdgePairGraph(pairs, num_vertices=num_vertices)

  @staticmethod
  def from_adj_matrix(adj, weighted=True):
    assert weighted, 'Unweighted graphs are NYI'
    if ss.issparse(adj):
      return SparseAdjacencyMatrixGraph(adj)
    return DenseAdjacencyMatrixGraph(adj)


class EdgePairGraph(Graph):
  def __init__(self, pairs, num_vertices=None):
    self._pairs = np.atleast_2d(pairs)
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
    row,col = self._pairs.T
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
    row,col = self._pairs.T
    diag_inds = row[np.equal(row,col)]
    to_add = np.arange(self._num_vertices)
    to_add = np.setdiff1d(to_add, diag_inds, assume_unique=True)
    if len(to_add) > 0:
      self._pairs = np.vstack((self._pairs, np.tile(to_add, (2,1)).T))
    return self

  def symmetrize(self, overwrite=True, method='sum'):
    '''Symmetrizes with the given method. {sum,max,avg}
    Returns a copy if overwrite=False.'''
    S = _symmetrize(self.matrix(dense=True, csr=True), method)
    if overwrite:
      self._pairs = np.transpose(np.nonzero(S))
      return self
    return Graph.from_adj_matrix(S)


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

  def symmetrize(self, overwrite=True, method='sum'):
    '''Symmetrizes with the given method. {sum,max,avg}
    Returns a copy if overwrite=False.'''
    S = _symmetrize(self._adj.tocsr(), method)
    if overwrite:
      self._adj = S
      return self
    return SparseAdjacencyMatrixGraph(S)


def _symmetrize(A, method):
  if method == 'sum':
    S = A + A.T
  elif method == 'max':
    if ss.issparse(A):
      S = A.maximum(A.T)
    else:
      S = np.maximum(A, A.T)
  else:
    S = (A + A.T) / 2.0
  return S
