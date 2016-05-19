from __future__ import division, absolute_import, print_function
import numpy as np
import scipy.sparse.csgraph as ssc


class TransformMixin(object):

  def kernelize(self, kernel):
    if kernel == 'none':
      return self
    if kernel == 'binary':
      if self.is_weighted():
        return self._update_edges(1, copy=True)
      return self
    if kernel == 'rbf':
      w = self.edge_weights()
      r = np.exp(-w / w.std())
      return self._update_edges(r, copy=True)
    raise ValueError('Invalid kernel type: %r' % kernel)

  def shortest_path_subtree(self, start_idx, directed=True):
    '''Returns a subgraph containing only the shortest paths from start_idx to
       every other vertex.
    '''
    adj = self.matrix()
    _, pred = ssc.dijkstra(adj, directed=directed, indices=start_idx,
                           return_predecessors=True)
    adj = ssc.reconstruct_path(adj, pred, directed=directed)
    if not directed:
      adj = adj + adj.T
    return self.__class__.from_adj_matrix(adj)

  def minimum_spanning_subtree(self):
    '''Returns the (undirected) minimum spanning tree subgraph.'''
    dist = self.matrix(dense=True, copy=True)
    dist[dist==0] = np.inf
    np.fill_diagonal(dist, 0)
    mst = ssc.minimum_spanning_tree(dist)
    return self.__class__.from_adj_matrix(mst + mst.T)
