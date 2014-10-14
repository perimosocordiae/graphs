import numpy as np
import scipy.sparse as ss

__all__ = [
    'Graph', 'EdgePairGraph', 'DenseAdjacencyMatrixGraph',
    'SparseAdjacencyMatrixGraph'
]


class Graph(object):

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

  def is_weighted(self):
    return False

  def is_directed(self):
    return True

  def adj_list(self):
    '''Generates a sequence of lists of neighbor indices:
        an adjacency list representation.'''
    adj = self.asmatrix(dense=True, csr=True)
    for row in adj:
      yield row.nonzero()[-1]

  def symmetrize(self, overwrite=True, method='sum'):
    '''Symmetrizes with the given method. {sum,max,avg}
    Returns a copy if overwrite=False.'''
    assert not overwrite
    A = self.asmatrix(dense=True, csr=True)
    if method == 'sum':
      S = A + A.T
    elif method == 'max':
      S = np.maximum(A, A.T)
    else:
      S = (A + A.T) / 2
    return Graph(matrix=S)

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
    W = self._adj[self.pairs()]
    if copy:
      return W.copy()
    return W

  def num_edges(self):
    return np.count_nonzero(self._adj)

  def add_self_edges(self, weight=1):
    '''Adds all i->i edges, in-place.'''
    np.fill_diagonal(self._adj, weight)
    return self


class SparseAdjacencyMatrixGraph(AdjacencyMatrixGraph):
  def __init__(self, adj):
    assert ss.issparse(adj)
    self._adj = adj
    assert self._adj.shape[0] == self._adj.shape[1]

  def matrix(self, copy=False, **kwargs):
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
    if copy:
      return W.copy()
    return W

  def num_edges(self):
    return self._adj.nnz

  def add_self_edges(self, weight=1):
    '''Adds all i->i edges, in-place.'''
    self._adj.setdiag(weight)
    return self
