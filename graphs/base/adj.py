import numpy as np
import scipy.sparse as ss

from base import Graph


class AdjacencyMatrixGraph(Graph):

  def pairs(self, copy=False):
    return np.transpose(np.nonzero(self._adj))

  def copy(self):
    return self.__class__(self._adj.copy())

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

  def add_edges(self, from_idx, to_idx, weight=1, symmetric=False, copy=False):
    '''Adds all from->to edges. weight may be a scalar or 1d array.
    If symmetric=True, also adds to->from edges with the same weights.'''
    weight = np.atleast_1d(1 if weight is None else weight)
    res_dtype = np.promote_types(weight.dtype, self._adj.dtype)
    adj = self._adj.astype(res_dtype, copy=copy)
    adj[from_idx, to_idx] = weight
    if symmetric:
      adj[to_idx, from_idx] = weight
    if copy:
      return DenseAdjacencyMatrixGraph(adj)
    self._adj = adj
    return self

  def symmetrize(self, method='sum', copy=False):
    '''Symmetrizes with the given method \in {sum,max,avg}'''
    adj = _symmetrize(self._adj, method)
    if copy:
      return DenseAdjacencyMatrixGraph(adj)
    self._adj = adj
    return self


class SparseAdjacencyMatrixGraph(AdjacencyMatrixGraph):
  def __init__(self, adj):
    assert ss.issparse(adj), 'SparseAdjacencyMatrixGraph input must be sparse'
    self._adj = adj
    assert self._adj.shape[0] == self._adj.shape[1]
    # Things go wrong if we have explicit zeros in the graph.
    if adj.format in ('csr', 'csc'):
      self._adj.eliminate_zeros()

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
    # Also assumes no explicit zeros
    if copy:
      return W.copy()
    return W

  def num_edges(self):
    return self._adj.nnz

  def add_edges(self, from_idx, to_idx, weight=1, symmetric=False, copy=False):
    '''Adds all from->to edges. weight may be a scalar or 1d array.
    If symmetric=True, also adds to->from edges with the same weights.'''
    adj = self._weightable_adj(weight, copy)
    adj[from_idx, to_idx] = weight
    if symmetric:
      adj[to_idx, from_idx] = weight
    return self._post_weighting(adj, weight, copy)

  def add_self_edges(self, weight=1, copy=False):
    '''Adds all i->i edges. weight may be a scalar or 1d array.'''
    adj = self._weightable_adj(weight, copy)
    try:
      adj.setdiag(weight)
    except TypeError:
      # Older scipy doesn't support setdiag on everything.
      adj = adj.tocsr()
      adj.setdiag(weight)
    return self._post_weighting(adj, weight, copy)

  def reweight(self, weight, edges=None, copy=False):
    '''Replaces existing edge weights. weight may be a scalar or 1d array.
    edges is a mask or index array that specifies a subset of edges to modify'''
    adj = self._weightable_adj(weight, copy)
    if edges is None:
      adj.data[:] = weight
    else:
      adj.data[edges] = weight
    return self._post_weighting(adj, weight, copy)

  def _weightable_adj(self, weight, copy):
    weight = np.atleast_1d(weight)
    res_dtype = np.promote_types(weight.dtype, self._adj.dtype)
    if copy or res_dtype is not self._adj.dtype:
      return self._adj.astype(res_dtype)
    return self._adj

  def _post_weighting(self, adj, weight, copy):
    if np.any(weight == 0):
      # TODO: be smarter about avoiding writing explicit zeros
      # We changed the sparsity structure, possibly.
      assert hasattr(adj, 'eliminate_zeros'), 'Other formats NYI'
      adj.eliminate_zeros()
    if copy:
      return SparseAdjacencyMatrixGraph(adj)
    self._adj = adj
    return self

  def symmetrize(self, method='sum', copy=False):
    '''Symmetrizes with the given method. {sum,max,avg}
    Returns a copy if overwrite=False.'''
    adj = _symmetrize(self._adj.tocsr(), method)
    if copy:
      return SparseAdjacencyMatrixGraph(adj)
    self._adj = adj
    return self


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
