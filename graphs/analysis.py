from __future__ import division
from itertools import count
import numpy as np
import scipy.sparse.csgraph as ssc
import scipy.sparse as ss

__all__ = [
    'connected_components', 'laplacian', 'shortest_path', 'greedy_coloring',
    'ave_laplacian', 'directed_laplacian', 'edge_traffic', 'bottlenecks'
]


# scipy.sparse.csgraph wrappers
def connected_components(G, **kwargs):
  return ssc.connected_components(G.matrix(), **kwargs)


def laplacian(G, **kwargs):
  return ssc.laplacian(G.matrix(), **kwargs)


def shortest_path(G, **kwargs):
  return ssc.shortest_path(G.matrix(), **kwargs)


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
    print "Warning: phi failed to converge"
  # L = Phi - (Phi P + P' Phi)/2
  return np.diag(phi) - ((phi * P.T).T + P.T * phi)/2


def edge_traffic(G, directed=False):
  adj = G.matrix(dense=True, lil=True, csr=True, csc=True)
  D, pred = ssc.shortest_path(adj, return_predecessors=True, directed=directed)
  counts = np.zeros_like(D, dtype=int)
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
        counts[k,j] += 1
        j = k
  return counts


def bottlenecks(G, n=1, directed=False, counts=None):
  """Finds n bottleneck edges, ranked by all-pairs path traffic."""
  if counts is None:
    counts = edge_traffic(G, directed=directed)
  edges = ss.dok_matrix(counts)
  top_k = np.argpartition(np.array(edges.values()), n-1)[:n]
  return np.array(edges.keys())[top_k]
