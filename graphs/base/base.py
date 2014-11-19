import numpy as np
import scipy.sparse as ss


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

  def add_edges(self, from_idx, to_idx, weight=None, symmetric=False):
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

  def to_igraph(self):
    '''Converts this Graph object to an igraph-compatible object.
    Requires the python-igraph library.'''
    # Import here to avoid ImportErrors when igraph isn't available.
    import igraph
    return igraph.Graph(n=self.num_vertices(), edges=self.pairs().tolist(),
                        directed=True)


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
