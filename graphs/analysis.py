from __future__ import division, absolute_import, print_function
from collections import defaultdict
from itertools import count
import numpy as np
import scipy.sparse.csgraph as ssc
import warnings
from .mini_six import range


class AnalysisMixin(object):

  # scipy.sparse.csgraph wrappers
  def connected_components(self, **kwargs):
    '''Mirrors the scipy.sparse.csgraph function of the same name:
    connected_components(G, directed=True, connection='weak',
                         return_labels=True)
    '''
    return ssc.connected_components(self.matrix(), **kwargs)

  def laplacian(self, **kwargs):
    '''Mirrors the scipy.sparse.csgraph function of the same name:
    laplacian(G, normed=False, return_diag=False, use_out_degree=False)
    '''
    return ssc.laplacian(self.matrix(), **kwargs)

  def shortest_path(self, **kwargs):
    '''Mirrors the scipy.sparse.csgraph function of the same name:
    shortest_path(G, method='auto', directed=True, return_predecessors=False,
                  unweighted=False, overwrite=False)
    '''
    # ssc.shortest_path requires one of these formats:
    adj = self.matrix(dense=True, csr=True, csc=True)
    return ssc.shortest_path(adj, **kwargs)

  def greedy_coloring(self):
    '''Returns a greedy vertex coloring, as an array of ints.'''
    n = self.num_vertices()
    coloring = np.zeros(n, dtype=int)
    for i, nbrs in enumerate(self.adj_list()):
      nbr_colors = set(coloring[nbrs])
      for c in count(1):
        if c not in nbr_colors:
          coloring[i] = c
          break
    return coloring

  def ave_laplacian(self):
    '''Another kind of laplacian normalization, used in the matlab PVF code.
    Uses the formula: L = I - D^{-1} * W'''
    W = self.matrix(dense=True)
    # calculate -inv(D)
    Dinv = W.sum(axis=0)
    mask = Dinv!=0
    Dinv[mask] = -1./Dinv[mask]
    # calculate -inv(D) * W
    lap = (Dinv * W.T).T
    # add I
    lap.flat[::W.shape[0]+1] += 1
    # symmetrize
    return (lap + lap.T) / 2.0

  def directed_laplacian(self, D=None, eta=0.99, tol=1e-12, max_iter=500):
    '''Computes the directed combinatorial graph laplacian.
    http://www-all.cs.umass.edu/pubs/2007/johns_m_ICML07.pdf

    D: (optional) N-array of degrees
    eta: probability of not teleporting (see the paper)
    tol, max_iter: convergence params for Perron vector calculation
    '''
    W = self.matrix(dense=True)
    n = W.shape[0]
    if D is None:
      D = W.sum(axis=1)
    # compute probability transition matrix
    with np.errstate(invalid='ignore', divide='ignore'):
      P = W.astype(float) / D[:,None]
    P[D==0] = 0
    # start at the uniform distribution Perron vector (phi)
    old_phi = np.ones(n) / n
    # iterate to the fixed point (teleporting random walk)
    for _ in range(max_iter):
      phi = eta * old_phi.dot(P) + (1-eta)/n
      if np.abs(phi - old_phi).max() < tol:
        break
      old_phi = phi
    else:
      warnings.warn("phi failed to converge after %d iterations" % max_iter)
    # L = Phi - (Phi P + P' Phi)/2
    return np.diag(phi) - ((phi * P.T).T + P.T * phi)/2

  def bandwidth(self):
    """Computes the 'bandwidth' of a graph."""
    return np.abs(np.diff(self.pairs(), axis=1)).max()

  def profile(self):
    """Measure of bandedness, also known as 'envelope size'."""
    leftmost_idx = np.argmax(self.matrix(dense=True).astype(bool), axis=0)
    return (np.arange(self.num_vertices()) - leftmost_idx).sum()

  def betweenness(self, kind='vertex', directed=None, weighted=None):
    '''Computes the betweenness centrality of a graph.
    kind : string, either 'vertex' (default) or 'edge'
    directed : bool, defaults to self.is_directed()
    weighted : bool, defaults to self.is_weighted()

    Note: When igraph is available, its implementation will be used for speed.
    '''
    assert kind in ('vertex', 'edge'), 'Invalid kind argument: ' + kind
    weighted = weighted is not False and self.is_weighted()
    d = directed if directed is not None else self.is_directed()
    try:
      ig = self.to_igraph()
    except ImportError:
      # igraph is not available, fall back to slower version
      dist, pred = self.shortest_path(directed=d, unweighted=(not weighted),
                                      return_predecessors=True)
      if kind == 'vertex':
        btw = _vertex_betweenness(dist, pred)
      else:
        btw = _edge_betweenness(dist, pred)
    else:
      w = self.edge_weights() if weighted else None
      if kind == 'vertex':
        btw = ig.betweenness(weights=w, directed=d)
      else:
        btw = ig.edge_betweenness(weights=w, directed=d)
    return np.array(btw)

  def eccentricity(self, directed=None, weighted=None):
    '''Maximum distance from each vertex to any other vertex.'''
    d = directed if directed is not None else self.is_directed()
    w = weighted if weighted is not None else self.is_weighted()
    sp = self.shortest_path(directed=d, unweighted=(not w))
    return sp.max(axis=0)

  def diameter(self, directed=None, weighted=None):
    '''Finds the length of the longest shortest path,
    a.k.a. the maximum graph eccentricity.'''
    return self.eccentricity(directed, weighted).max()

  def radius(self, directed=None, weighted=None):
    '''minimum graph eccentricity'''
    return self.eccentricity(directed, weighted).min()


def _vertex_betweenness(dist, pred):
  btw = np.zeros(dist.shape[0])
  ii, jj = np.nonzero((~np.isinf(dist)) & (dist > 0))
  for i,j in zip(ii, jj):
    v = pred[i, j]
    while v != i:
      btw[v] += 1
      v = pred[i, v]
  return btw


def _edge_betweenness(dist, pred):
  btw = defaultdict(int)
  ii, jj = np.nonzero((~np.isinf(dist)) & (dist > 0))
  for i,j in zip(ii, jj):
    edge = (pred[i, j], j)
    while edge[0] != i:
      btw[edge] += 1
      edge = (pred[i, edge[0]], edge[0])
    btw[edge] += 1
  return [btw[k] for k in sorted(btw.keys())]
