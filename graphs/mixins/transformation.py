from __future__ import division, absolute_import, print_function
import numpy as np
import scipy.sparse as ss
import scipy.sparse.csgraph as ssc
from scipy.linalg import solve
from collections import deque

from ..mini_six import range


class TransformMixin(object):

  def kernelize(self, kernel):
    '''Re-weight according to a specified kernel function.
    kernel : str, {none, binary, rbf}
      none   -> no reweighting
      binary -> all edges are given weight 1
      rbf    -> applies a gaussian function to edge weights
    '''
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

  def barycenter_edge_weights(self, X, copy=True, reg=1e-3):
    '''Re-weight such that the sum of each vertex's edge weights is 1.
    The resulting weighted graph is suitable for locally linear embedding.
    reg : amount of regularization to keep the problem well-posed
    '''
    new_weights = []
    for i, adj in enumerate(self.adj_list()):
      C = X[adj] - X[i]
      G = C.dot(C.T)
      trace = np.trace(G)
      r = reg * trace if trace > 0 else reg
      G.flat[::G.shape[1] + 1] += r
      w = solve(G, np.ones(G.shape[0]), sym_pos=True,
                overwrite_a=True, overwrite_b=True)
      w /= w.sum()
      new_weights.extend(w.tolist())
    return self.reweight(new_weights, copy=copy)

  def connected_subgraphs(self, directed=True, ordered=False):
    '''Generates connected components as subgraphs.
    When ordered=True, subgraphs are ordered by number of vertices.
    '''
    num_ccs, labels = self.connected_components(directed=directed)
    # check the trivial case first
    if num_ccs == 1:
      yield self
      raise StopIteration
    if ordered:
      # sort by descending size (num vertices)
      order = np.argsort(np.bincount(labels))[::-1]
    else:
      order = range(num_ccs)

    # don't use self.subgraph() here, because we can reuse adj
    adj = self.matrix(dense=True, csr=True, csc=True)
    for c in order:
      mask = labels == c
      sub_adj = adj[mask][:,mask]
      yield self.__class__.from_adj_matrix(sub_adj)

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

  def neighborhood_subgraph(self, start_idx, radius=1, weighted=True,
                            directed=True, return_mask=False):
    '''Returns a subgraph containing only vertices within a given
       geodesic radius of start_idx.'''
    adj = self.matrix(dense=True, csr=True, csc=True)
    dist = ssc.dijkstra(adj, directed=directed, indices=start_idx,
                        unweighted=(not weighted), limit=radius)
    mask = np.isfinite(dist)
    sub_adj = adj[mask][:,mask]
    g = self.__class__.from_adj_matrix(sub_adj)
    if return_mask:
      return g, mask
    return g

  def circle_tear(self, spanning_tree='mst', cycle_len_thresh=5, spt_idx=None,
                  copy=True):
    '''Circular graph tearing.

    spanning_tree: one of {'mst', 'spt'}
    cycle_len_thresh: int, length of longest allowable cycle
    spt_idx: int, start vertex for shortest_path_subtree, random if None

    From "How to project 'circular' manifolds using geodesic distances?"
      by Lee & Verleysen, ESANN 2004.

    See also: shortest_path_subtree, minimum_spanning_subtree
    '''
    # make the initial spanning tree graph
    if spanning_tree == 'mst':
      tree = self.minimum_spanning_subtree().matrix()
    elif spanning_tree == 'spt':
      if spt_idx is None:
        spt_idx = np.random.choice(self.num_vertices())
      tree = self.shortest_path_subtree(spt_idx, directed=False).matrix()

    # find edges in self but not in the tree
    potential_edges = np.argwhere(ss.triu(self.matrix() - tree))

    # remove edges that induce large cycles
    ii, jj = _find_cycle_inducers(tree, potential_edges, cycle_len_thresh)
    return self.remove_edges(ii, jj, symmetric=True, copy=copy)

  def cycle_cut(self, cycle_len_thresh=12, directed=False, copy=True):
    '''CycleCut algorithm: removes bottleneck edges.
    Paper DOI: 10.1.1.225.5335
    '''
    symmetric = not directed
    adj = self.kernelize('binary').matrix(csr=True, dense=True, copy=True)
    if symmetric:
      adj = adj + adj.T

    removed_edges = []
    while True:
      c = _atomic_cycle(adj, cycle_len_thresh, directed=directed)
      if c is None:
        break
      # remove edges in the cycle
      ii, jj = c.T
      adj[ii,jj] = 0
      if symmetric:
        adj[jj,ii] = 0
      removed_edges.extend(c)

    #XXX: if _atomic_cycle changes, may need to do this on each loop
    if ss.issparse(adj):
      adj.eliminate_zeros()

    # select only the necessary cuts
    ii, jj = _find_cycle_inducers(adj, removed_edges, cycle_len_thresh,
                                  directed=directed)
    # remove the bad edges
    return self.remove_edges(ii, jj, symmetric=symmetric, copy=copy)


def _atomic_cycle(adj, length_thresh, directed=False):
  # TODO: make this more efficient
  start_vertex = np.random.choice(adj.shape[0])
  # run BFS
  q = deque([start_vertex])
  visited_vertices = set([start_vertex])
  visited_edges = set()
  while q:
    a = q.popleft()
    nbrs = adj[a].nonzero()[-1]
    for b in nbrs:
      if b not in visited_vertices:
        q.append(b)
        visited_vertices.add(b)
        visited_edges.add((a,b))
        if not directed:
          visited_edges.add((b,a))
        continue
      # run an inner BFS
      inner_q = deque([b])
      inner_visited = set([b])
      parent_vertices = {b: -1}
      while inner_q:
        c = inner_q.popleft()
        inner_nbrs = adj[c].nonzero()[-1]
        for d in inner_nbrs:
          if d in inner_visited or (d,c) not in visited_edges:
            continue
          parent_vertices[d] = c
          inner_q.append(d)
          inner_visited.add(d)
          if d != a:
            continue
          # atomic cycle found
          cycle = []
          while parent_vertices[d] != -1:
            x, d = d, parent_vertices[d]
            cycle.append((x, d))
          cycle.append((d, a))
          if len(cycle) >= length_thresh:
            return np.array(cycle)
          else:
            # abort the inner BFS
            inner_q.clear()
            break
      # finished considering edge a->b
      visited_edges.add((a,b))
      if not directed:
        visited_edges.add((b,a))
  # no cycles found
  return None


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
