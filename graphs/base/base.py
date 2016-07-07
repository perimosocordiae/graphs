from __future__ import absolute_import
import numpy as np
import scipy.sparse as ss
import warnings
from sklearn.metrics.pairwise import paired_distances

from ..mini_six import zip
from ..mixins import (
    AnalysisMixin, EmbedMixin, LabelMixin, TransformMixin, VizMixin)


class Graph(AnalysisMixin, EmbedMixin, LabelMixin, TransformMixin, VizMixin):

  def __init__(self, *args, **kwargs):
    raise NotImplementedError('Graph should not be instantiated directly')

  def pairs(self, copy=False, directed=True):
    '''Returns a (num_edges,2)-array of vertex indices (s,t).
    When directed=False, only pairs with s <= t are returned.'''
    raise NotImplementedError()

  def matrix(self, copy=False, **kwargs):
    '''Returns a (num_vertices,num_vertices) array or sparse matrix, M,
    where M[s,t] is the weight of edge (s,t).

    TODO: explain the kwargs situation.
    '''
    raise NotImplementedError()

  def edge_weights(self, copy=False, directed=True):
    '''Returns a (num_edges,)-array of edge weights.
    Weights correspond to the (s,t) pairs returned by pairs().
    When directed=False, only weights with s <= t are returned.'''
    raise NotImplementedError()

  def num_edges(self):
    raise NotImplementedError()

  def num_vertices(self):
    raise NotImplementedError()

  def symmetrize(self, method='sum', copy=False):
    '''Symmetrizes with the given method. {sum,max,avg}
    Returns a copy if overwrite=False.'''
    raise NotImplementedError()

  def add_edges(self, from_idx, to_idx, weight=1, symmetric=False, copy=False):
    '''Adds all from->to edges. weight may be a scalar or 1d array.
    If symmetric=True, also adds to->from edges with the same weights.'''
    raise NotImplementedError()

  def remove_edges(self, from_idx, to_idx, symmetric=False, copy=False):
    '''Removes all from->to edges, without making sure they already exist.
    If symmetric=True, also removes to->from edges.'''
    raise NotImplementedError()

  def _update_edges(self, weights, copy=False):
    raise NotImplementedError()

  def subgraph(self, mask):
    '''Returns the subgraph with vertices V[mask].
    mask : boolean mask, index, or slice'''
    raise NotImplementedError()

  def copy(self):
    raise NotImplementedError()

  def is_weighted(self):
    '''Returns True if edges have associated weights.'''
    return False

  def is_directed(self):
    '''Returns True if edges *may be* asymmetric.'''
    return True

  def add_self_edges(self, weight=None, copy=False):
    '''Adds all i->i edges. weight may be a scalar or 1d array.'''
    ii = np.arange(self.num_vertices())
    return self.add_edges(ii, ii, weight=weight, symmetric=False, copy=copy)

  def reweight(self, weight, edges=None, copy=False):
    '''Replaces existing edge weights. weight may be a scalar or 1d array.
    edges is a mask or index array that specifies a subset of edges to modify'''
    if not self.is_weighted():
      warnings.warn('Cannot supply weights for unweighted graph; '
                    'ignoring call to reweight')
      return self
    if edges is None:
      return self._update_edges(weight, copy=copy)
    ii, jj = self.pairs()[edges].T
    return self.add_edges(ii, jj, weight=weight, symmetric=False, copy=copy)

  def reweight_by_distance(self, coords, metric='l2', copy=False):
    '''Replaces existing edge weights by distances between connected vertices.
    The new weight of edge (i,j) is given by: metric(coords[i], coords[j]).
    coords : (num_vertices x d) array of coordinates, in vertex order
    metric : str or callable, see sklearn.metrics.pairwise.paired_distances'''
    if not self.is_weighted():
      warnings.warn('Cannot supply weights for unweighted graph; '
                    'ignoring call to reweight_by_distance')
      return self
    # TODO: take advantage of symmetry of metric function
    ii, jj = self.pairs().T
    d = paired_distances(coords[ii], coords[jj], metric=metric)
    return self._update_edges(d, copy=copy)

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

  def to_igraph(self, weighted=None):
    '''Converts this Graph object to an igraph-compatible object.
    Requires the python-igraph library.'''
    # Import here to avoid ImportErrors when igraph isn't available.
    import igraph
    ig = igraph.Graph(n=self.num_vertices(), edges=self.pairs().tolist(),
                      directed=self.is_directed())
    if weighted is not False and self.is_weighted():
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

  def to_networkx(self, directed=None):
    '''Converts this Graph object to a networkx-compatible object.
    Requires the networkx library.'''
    import networkx as nx
    directed = directed if directed is not None else self.is_directed()
    cls = nx.DiGraph if directed else nx.Graph
    adj = self.matrix()
    if ss.issparse(adj):
      return nx.from_scipy_sparse_matrix(adj, create_using=cls())
    return nx.from_numpy_matrix(adj, create_using=cls())
