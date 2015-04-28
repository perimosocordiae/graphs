import numpy as np
from graphs.analysis import AnalysisMixin
from graphs.embed import EmbedMixin
from graphs.viz import VizMixin


class Graph(AnalysisMixin, EmbedMixin, VizMixin):

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

  def symmetrize(self, method='sum', copy=False):
    raise NotImplementedError()

  def add_edges(self, from_idx, to_idx, weight=1, symmetric=False, copy=False):
    raise NotImplementedError()

  def copy(self):
    raise NotImplementedError()

  def is_weighted(self):
    '''Returns True if edges have associated weights.'''
    return False

  def is_directed(self):
    '''Returns True if edges *may be* non-symmetric.'''
    return True

  def add_self_edges(self, weight=None, copy=False):
    '''Adds all i->i edges. weight may be a scalar or 1d array.'''
    ii = np.arange(self.num_vertices())
    return self.add_edges(ii, ii, weight=weight, symmetric=False, copy=copy)

  def reweight(self, weight, edges=None, copy=False):
    '''Replaces existing edge weights. weight may be a scalar or 1d array.
    edges is a mask or index array that specifies a subset of edges to modify'''
    P = self.pairs()
    if edges is not None:
      P = P[edges]
    return self.add_edges(*P.T, weight=weight, symmetric=False, copy=copy)

  def adj_list(self):
    '''Generates a sequence of lists of neighbor indices:
        an adjacency list representation.'''
    adj = self.matrix(dense=True, csr=True)
    for row in adj:
      yield row.nonzero()[-1]

  def degree(self, kind='out', weighted=True):
    '''Returns an array of vertex degrees.
    kind : either 'in' or 'out', useful for directed graphs
    weighted : controls whether to count edges or sum their weights
    '''
    axis = 1 if kind == 'out' else 0
    adj = self.matrix(dense=True, csr=1-axis, csc=axis)
    if not weighted and self.is_weighted():
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
    ig = igraph.Graph(n=self.num_vertices(), edges=self.pairs().tolist(),
                      directed=self.is_directed())
    if self.is_weighted():
      ig.es['weight'] = self.edge_weights()
    return ig

  def to_graph_tool(self):
    '''Converts this Graph object to a graph_tool-compatible object.
    Requires the graph_tool library.
    Note that the internal ordering of graph_tool seems to be column-major.'''
    # Import here to avoid ImportErrors when graph_tool isn't available.
    import graph_tool
    gt = graph_tool.Graph(directed=self.is_directed())
    gt.add_edge_list(self.pairs())
    if self.is_weighted():
      weights = gt.new_edge_property('double')
      for e,w in zip(gt.edges(), self.edge_weights()):
        weights[e] = w
      gt.edge_properties['weight'] = weights
    return gt
