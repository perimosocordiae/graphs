from __future__ import division, absolute_import, print_function
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree


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
    dist, pred = self.shortest_path(return_predecessors=True, directed=directed)
    ii = pred[start_idx]
    mask = ii >= 0
    jj, = np.where(mask)
    ii = ii[mask]
    if not directed:
      ii, jj = np.concatenate((ii, jj)), np.concatenate((jj, ii))
    adj = coo_matrix((dist[ii,jj], (ii, jj)), shape=dist.shape)
    return self.__class__.from_adj_matrix(adj)

  def minimum_spanning_subtree(self):
    '''Returns the (undirected) minimum spanning tree subgraph.'''
    dist = self.matrix(dense=True, copy=True)
    dist[dist==0] = np.inf
    np.fill_diagonal(dist, 0)
    # symmetrize (TODO: might not be necessary?)
    dist = np.minimum(dist, dist.T)
    mst = minimum_spanning_tree(dist)
    return self.__class__.from_adj_matrix(mst + mst.T)
