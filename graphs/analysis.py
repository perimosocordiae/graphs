from __future__ import division
from collections import Counter
from itertools import count
import numpy as np
import scipy.sparse.csgraph as ssc
import warnings

__all__ = [
    'connected_components', 'laplacian', 'shortest_path', 'greedy_coloring',
    'ave_laplacian', 'directed_laplacian', 'edge_traffic', 'bottlenecks',
    'bandwidth', 'profile'
]


# scipy.sparse.csgraph wrappers
def connected_components(G, **kwargs):
  '''Mirrors the scipy.sparse.csgraph function of the same name:
  connected_components(G, directed=True, connection='weak', return_labels=True)
  '''
  return ssc.connected_components(G.matrix(), **kwargs)


def laplacian(G, **kwargs):
  '''Mirrors the scipy.sparse.csgraph function of the same name:
  laplacian(G, normed=False, return_diag=False, use_out_degree=False)
  '''
  return ssc.laplacian(G.matrix(), **kwargs)


def shortest_path(G, **kwargs):
  '''Mirrors the scipy.sparse.csgraph function of the same name:
  shortest_path(G, method='auto', directed=True, return_predecessors=False,
                unweighted=False, overwrite=False)
  '''
  # ssc.shortest_path requires one of these formats:
  adj = G.matrix(dense=True, lil=True, csr=True, csc=True)
  return ssc.shortest_path(adj, **kwargs)


def greedy_coloring(G):
  '''Returns a greedy vertex coloring, as an array of ints.'''
  n = G.num_vertices()
  coloring = np.zeros(n, dtype=int)
  for i, nbrs in enumerate(G.adj_list()):
    nbr_colors = set(coloring[nbrs])
    for c in count(start=1):
      if c not in nbr_colors:
        coloring[i] = c
        break
  return coloring


def ave_laplacian(G):
  '''Another kind of laplacian normalization, used in the matlab PVF code.
  Uses the formula: L = I - D^{-1} * W'''
  W = G.matrix(dense=True)
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


def directed_laplacian(G, D=None, eta=0.99, tol=1e-12, max_iter=500):
  '''Computes the directed combinatorial graph laplacian.
  http://www-all.cs.umass.edu/pubs/2007/johns_m_ICML07.pdf

  D: (optional) N-array of degrees
  eta: probability of not teleporting (see the paper)
  tol, max_iter: convergence params for Perron vector calculation
  '''
  W = G.matrix(dense=True)
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
  for _ in xrange(max_iter):
    phi = eta * old_phi.dot(P) + (1-eta)/n
    if np.abs(phi - old_phi).max() < tol:
      break
    old_phi = phi
  else:
    warnings.warn("phi failed to converge after %d iterations" % max_iter)
  # L = Phi - (Phi P + P' Phi)/2
  return np.diag(phi) - ((phi * P.T).T + P.T * phi)/2


def edge_traffic(G, directed=False):
  """Counts number of shortest paths that use a given edge.
  Returns a dictionary of (ei,ej) -> # of paths
  """
  D, pred = shortest_path(G, return_predecessors=True, directed=directed)
  counts = Counter()
  n = D.shape[0]
  if directed:
    j_range = lambda i: xrange(n)
  else:
    j_range = lambda i: xrange(i+1, n)
  for i in xrange(n):
    pp = pred[i]
    for j in j_range(i):
      if np.isinf(D[i,j]):
        continue
      while j != i:
        k = pp[j]
        counts[(k,j)] += 1
        j = k
  return counts


def bottlenecks(G, n=1, directed=False, return_counts=False):
  """Finds n bottleneck edges, ranked by all-pairs path traffic."""
  traffic = edge_traffic(G, directed=directed)
  top_edges = np.empty((n, 2), dtype=int)
  if return_counts:
    top_counts = np.empty(n, dtype=int)
    for i, (edge, c) in enumerate(traffic.most_common(n)):
      top_edges[i] = edge
      top_counts[i] = c
    return top_edges, top_counts
  for i, (edge, c) in enumerate(traffic.most_common(n)):
    top_edges[i] = edge
  return top_edges


def bandwidth(G):
  """Computes the 'bandwidth' of a graph."""
  return np.abs(np.diff(G.pairs(), axis=1)).max()


def profile(G):
  """Measure of bandedness, also known as 'envelope size'."""
  leftmost_idx = np.argmax(G.matrix(dense=True).astype(bool), axis=0)
  return (np.arange(G.num_vertices()) - leftmost_idx).sum()
