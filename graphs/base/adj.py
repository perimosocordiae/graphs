from __future__ import absolute_import

import numpy as np
import scipy.sparse as ss

from .base import Graph


class AdjacencyMatrixGraph(Graph):

  def copy(self):
    return self.__class__(self._adj.copy())

  def num_vertices(self):
    return self._adj.shape[0]

  def is_weighted(self):
    return True

  def subgraph(self, mask):
    adj = self.matrix(dense=True, csr=True, csc=True)
    sub_adj = adj[mask][:,mask]
    return Graph.from_adj_matrix(sub_adj)

  subgraph.__doc__ = Graph.subgraph.__doc__


class DenseAdjacencyMatrixGraph(AdjacencyMatrixGraph):
  def __init__(self, adj):
    self._adj = np.atleast_2d(adj)
    assert self._adj.shape[0] == self._adj.shape[1]

  def pairs(self, copy=False, directed=True):
    adj = self._adj if directed else np.triu(self._adj)
    return np.transpose(np.nonzero(adj))

  def matrix(self, copy=False, **kwargs):
    if not kwargs or 'dense' in kwargs:
      if copy:
        return self._adj.copy()
      return self._adj
    if 'csr' in kwargs:
      return ss.csr_matrix(self._adj)
    raise NotImplementedError('Unknown matrix type(s): %s' % (
                              tuple(kwargs.keys()),))

  def edge_weights(self, copy=False, directed=True):
    ii,jj = self.pairs(directed=directed).T
    return self._adj[ii,jj]

  def num_edges(self):
    return np.count_nonzero(self._adj)

  def add_edges(self, from_idx, to_idx, weight=1, symmetric=False, copy=False):
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

  def remove_edges(self, from_idx, to_idx, symmetric=False, copy=False):
    adj = self._adj.copy() if copy else self._adj
    adj[from_idx, to_idx] = 0
    if symmetric:
      adj[to_idx, from_idx] = 0
    if copy:
      return DenseAdjacencyMatrixGraph(adj)
    self._adj = adj
    return self

  def _update_edges(self, weights, copy=False):
    weights = np.asarray(weights)
    res_dtype = np.promote_types(weights.dtype, self._adj.dtype)
    adj = self._adj.astype(res_dtype, copy=copy)
    adj[adj != 0] = weights
    if copy:
      return DenseAdjacencyMatrixGraph(adj)
    self._adj = adj
    return self

  def symmetrize(self, method='sum', copy=False):
    adj = _symmetrize(self._adj, method)
    if copy:
      return DenseAdjacencyMatrixGraph(adj)
    self._adj = adj
    return self

  pairs.__doc__ = Graph.pairs.__doc__
  matrix.__doc__ = Graph.matrix.__doc__
  edge_weights.__doc__ = Graph.edge_weights.__doc__
  add_edges.__doc__ = Graph.add_edges.__doc__
  remove_edges.__doc__ = Graph.remove_edges.__doc__
  symmetrize.__doc__ = Graph.symmetrize.__doc__


class SparseAdjacencyMatrixGraph(AdjacencyMatrixGraph):
  def __init__(self, adj, may_have_zeros=True):
    assert ss.issparse(adj), 'SparseAdjacencyMatrixGraph input must be sparse'
    if adj.format not in ('coo', 'csr', 'csc'):
      adj = adj.tocsr()
    self._adj = adj
    assert self._adj.shape[0] == self._adj.shape[1]
    if may_have_zeros:
      # Things go wrong if we have explicit zeros in the graph.
      _eliminate_zeros(self._adj)

  def pairs(self, copy=False, directed=True):
    adj = self._adj if directed else ss.triu(self._adj)
    return np.transpose(adj.nonzero())

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
    raise NotImplementedError('Unknown matrix type(s): %s' % (
                              tuple(kwargs.keys()),))

  def edge_weights(self, copy=False, directed=True):
    if not directed:
      ii, jj = ss.triu(self._adj).nonzero()
      return np.asarray(self._adj[ii, jj]).ravel()
    # XXX: assumes correct internal ordering and no explicit zeros
    w = self._adj.data.ravel()
    if copy:
      return w.copy()
    return w

  def num_edges(self):
    return self._adj.nnz

  def add_edges(self, from_idx, to_idx, weight=1, symmetric=False, copy=False):
    adj = self._weightable_adj(weight, copy)
    if adj.format == 'coo':
      adj = adj.tocsr()
    adj[from_idx, to_idx] = weight
    if symmetric:
      adj[to_idx, from_idx] = weight
    return self._post_weighting(adj, weight, copy)

  def remove_edges(self, from_idx, to_idx, symmetric=False, copy=False):
    adj = self._adj.copy() if copy else self._adj
    if adj.format == 'coo':
      adj = adj.tocsr()
    adj[from_idx, to_idx] = 0
    if symmetric:
      adj[to_idx, from_idx] = 0
    return self._post_weighting(adj, 0, copy)

  def _update_edges(self, weights, copy=False):
    adj = self._weightable_adj(weights, copy)
    adj.data[:] = weights
    return self._post_weighting(adj, weights, copy)

  def add_self_edges(self, weight=1, copy=False):
    adj = self._weightable_adj(weight, copy)
    try:
      adj.setdiag(weight)
    except TypeError:  # pragma: no cover
      # Older scipy doesn't support setdiag on everything.
      adj = adj.tocsr()
      adj.setdiag(weight)
    return self._post_weighting(adj, weight, copy)

  def reweight(self, weight, edges=None, copy=False):
    adj = self._weightable_adj(weight, copy)
    if edges is None:
      adj.data[:] = weight
    else:
      adj.data[edges] = weight
    return self._post_weighting(adj, weight, copy)

  def _weightable_adj(self, weight, copy):
    weight = np.atleast_1d(weight)
    adj = self._adj
    res_dtype = np.promote_types(weight.dtype, adj.dtype)
    if copy:
      adj = adj.copy()
    if res_dtype is not adj.dtype:
      adj.data = adj.data.astype(res_dtype)
    return adj

  def _post_weighting(self, adj, weight, copy):
    # Check if we might have changed the sparsity structure by adding zeros
    has_zeros = np.any(weight == 0)
    if copy:
      return SparseAdjacencyMatrixGraph(adj, may_have_zeros=has_zeros)
    self._adj = _eliminate_zeros(adj) if has_zeros else adj
    return self

  def symmetrize(self, method='sum', copy=False):
    adj = _symmetrize(self._adj.tocsr(), method)
    if copy:
      return SparseAdjacencyMatrixGraph(adj, may_have_zeros=False)
    self._adj = adj
    return self

  pairs.__doc__ = Graph.pairs.__doc__
  matrix.__doc__ = Graph.matrix.__doc__
  edge_weights.__doc__ = Graph.edge_weights.__doc__
  add_edges.__doc__ = Graph.add_edges.__doc__
  remove_edges.__doc__ = Graph.remove_edges.__doc__
  symmetrize.__doc__ = Graph.symmetrize.__doc__
  add_self_edges.__doc__ = Graph.add_self_edges.__doc__
  reweight.__doc__ = Graph.reweight.__doc__


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


def _eliminate_zeros(A):
  if hasattr(A, 'eliminate_zeros'):
    A.eliminate_zeros()
  elif A.format == 'coo':  # pragma: no cover
    # old scipy doesn't provide coo_matrix.eliminate_zeros
    nz_mask = A.data != 0
    A.data = A.data[nz_mask]
    A.row = A.row[nz_mask]
    A.col = A.col[nz_mask]
  else:
    raise ValueError("Can't eliminate_zeros from type: %s" % type(A))
  return A
