from __future__ import division, absolute_import, print_function
import numpy as np
import scipy.sparse as ss
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

  def circle_tear(self, spanning_tree='mst', cycle_len_thresh=5, idx=None):
    '''Circular graph tearing.

    spanning_tree: one of {'mst', 'spt'}
    cycle_len_thresh: int, length of longest allowable cycle
    idx: int, start vertex for shortest_path_subtree, random if None

    From "How to project 'circular' manifolds using geodesic distances?"
      by Lee & Verleysen, ESANN 2004.

    See also: shortest_path_subtree, minimum_spanning_subtree
    '''
    # make the initial spanning tree graph
    if spanning_tree == 'mst':
      tree = self.minimum_spanning_subtree().matrix()
    elif spanning_tree == 'spt':
      if idx is None:
        idx = np.random.choice(self.num_vertices())
      tree = self.shortest_path_subtree(idx, directed=False).matrix()

    # find edges in self but not in the tree
    potential_edges = np.argwhere(ss.triu(self.matrix() - tree))

    # remove edges that induce large cycles
    ii, jj = _find_cycle_inducers(tree, potential_edges, cycle_len_thresh)
    return self.remove_edges(ii, jj, symmetric=True, copy=True)


def _find_cycle_inducers(adj, potential_edges, length_thresh, directed=False):
    # remove edges that induce large cycles
    path_dist = ssc.dijkstra(adj, directed=directed, return_predecessors=False,
                             unweighted=True)
    remove_ii, remove_jj = [], []
    for i,j in potential_edges:
      if length_thresh < path_dist[i,j] < np.inf:
        remove_ii.append(i)
        remove_jj.append(j)
      else:
        # keeping this edge: update path lengths
        tmp = (path_dist[:,i] + 1)[:,None] + path_dist[j,:]
        ii, jj = np.nonzero(tmp < path_dist)
        new_lengths = tmp[ii, jj]
        path_dist[ii,jj] = new_lengths
        if not directed:
          path_dist[jj,ii] = new_lengths
    return remove_ii, remove_jj
