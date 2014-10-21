from collections import deque
import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse.csgraph as ssc
from matplotlib import pyplot as plt

'''Sparse symmetric matrix reordering to reduce bandwidth/diagonalness.
Methods:
 - cuthill_mckee
 - node_centroid_hill_climbing
 - maximum_bipartite_matching (with recent scipy)
References:
 - ftp://ftp.numerical.rl.ac.uk/pub/talks/jas.ala06.24VII06.pdf
 - http://www.jstor.org/stable/2156090 (profile defn, NYI RCM improvements)
 - https://www.cs.purdue.edu/homes/apothen/env3.pdf (laplacian, NYI sloan alg)
'''


# Sparse versions.
def _cuthill_mckee(G, degree=None):
  n = G.shape[0]
  queue = deque([])
  result = []
  if degree is None:
    degree = G.sum(axis=0)
  remaining = dict(enumerate(degree))
  while len(result) != n:
    queue.append(min(remaining, key=remaining.get))
    while queue:
      p = queue.popleft()
      if p not in remaining:
        continue
      result.append(p)
      del remaining[p]
      nbrs = [c for c in np.where(G[p])[0] if c in remaining]
      queue.extend(sorted(nbrs, key=remaining.get))
  return np.array(result)

if hasattr(ssc, 'reverse_cuthill_mckee'):
  def cuthill_mckee(G, degree=None):
    sG = csr_matrix(G)
    return ssc.reverse_cuthill_mckee(sG, symmetric_mode=True)
else:
  cuthill_mckee = _cuthill_mckee


def laplacian_reordering(G):
  L = ssc.laplacian(G)
  vals, vecs = np.linalg.eigh(L)
  vec, = vecs[vals == vals[vals>0].min()]
  return np.argsort(vec)


def bandwidth(X, order=None):
  if order is not None:
    X = X[np.ix_(order, order)]
  r,c = np.nonzero(X)
  return np.abs(r - c).max()


def profile(X, order=None):
  '''Measure of bandedness, also known as 'envelope size'.'''
  if order is not None:
    X = X[np.ix_(order, order)]
  leftmost_idx = np.argmax(X.astype(bool), axis=0)
  return (np.arange(X.shape[0]) - leftmost_idx).sum()


def node_centroid_hill_climbing(G, relax=1, num_centerings=20, verbose=False):
  # Initialize order with BFS from a random start node.
  order = breadth_first_order(G)
  for it in xrange(num_centerings):
    B = bandwidth(G, order)
    nc_order = node_center(G, order, relax=relax)
    nc_B = bandwidth(G, nc_order)
    if nc_B < B:
      if verbose:
        print 'post-center', B, nc_B
      order = nc_order
    order = hill_climbing(G, order, verbose=verbose)
  return order


def breadth_first_order(G):
  inds = np.arange(G.shape[0])
  total_order = []
  while len(inds) > 0:
    order = ssc.breadth_first_order(G, np.random.choice(inds),
                                    return_predecessors=False)
    inds = np.setdiff1d(inds, order, assume_unique=True)
    total_order = np.append(total_order, order)
  return total_order.astype(int)


def critical_vertices(G, order, relax=1, bw=None):
  go = G[np.ix_(order,order)]
  if bw is None:
    bw = bandwidth(G, order)
  if relax == 1:
    for i in np.where(np.diag(go, -bw))[0]:
      yield bw + i, i
  else:
    crit = relax * bw
    for u, v in np.transpose(np.where(np.tril(go, -np.floor(crit)))):
      if np.abs(u-v) >= crit:
        yield u, v


def node_center(G, order, relax=0.99):
  weights = order.copy().astype(float)
  counts = np.ones_like(order)
  inv_order = np.argsort(order)
  for i, j in critical_vertices(G, order, relax):
    u = inv_order[i]
    v = inv_order[j]
    weights[u] += j  # order[v]
    counts[u] += 1
    weights[v] += i  # order[u]
    counts[v] += 1
  weights /= counts
  return np.argsort(weights)


def hill_climbing(G, order, verbose=False):
  B = bandwidth(G, order)
  while True:
    inv_order = np.argsort(order)
    for i, j in critical_vertices(G, order, bw=B):
      u = inv_order[i]
      v = inv_order[j]
      for w,k in enumerate(order):
        if not (k < i or k > j):
          continue
        new_order = order.copy()
        if k < i:
          new_order[[u,w]] = new_order[[w,u]]
        elif k > j:
          new_order[[v,w]] = new_order[[w,v]]

        new_B = bandwidth(G, new_order)
        if new_B < B:
          order = new_order
          if verbose:
            print 'improved B', B, new_B
          B = new_B
          break
        elif new_B == B:
          nc = sum(1 for _ in critical_vertices(G, order, bw=B))
          new_nc = sum(1 for _ in critical_vertices(G, new_order, bw=B))
          if new_nc < nc:
            order = new_order
            if verbose:
              print 'improved nc', nc, new_nc
            break
      else:
        continue
      break
    else:
      break
  return order


def diagonalness(X):
  '''Rough measure of how banded X is. Penalizes values near diagonal.'''
  nX = X / np.linalg.norm(X)
  i,j = np.triu_indices_from(X)
  D = np.zeros_like(X)
  D[i,j] = np.abs(i-j)
  D += D.T
  D /= D.max()
  return np.sum(nX*(1-D))


def main_sparse(G):
  cm_order = cuthill_mckee(G)
  nchc_order = node_centroid_hill_climbing(G)
  lap_order = laplacian_reordering(G)

  print 'original', bandwidth(G), profile(G)
  print 'cuthill_mckee', bandwidth(G, cm_order), profile(G, cm_order)
  print 'NCHC', bandwidth(G, nchc_order), profile(G, nchc_order)
  print 'laplacian_reordering', bandwidth(G, lap_order), profile(G, lap_order)

  _,axes = plt.subplots(ncols=2,nrows=2)
  axes[0,0].spy(G)
  axes[0,0].set_title('original')
  axes[0,1].spy(G[np.ix_(cm_order,cm_order)])
  axes[0,1].set_title('cuthill_mckee')
  axes[1,0].spy(G[np.ix_(nchc_order,nchc_order)])
  axes[1,0].set_title('NCHC')
  axes[1,1].spy(G[np.ix_(lap_order,lap_order)])
  axes[1,1].set_title('laplacian_reordering')
  plt.setp([ax.get_xticklabels() for ax in axes.flat], visible=False)
  plt.setp([ax.get_yticklabels() for ax in axes.flat], visible=False)
  plt.show()


def test_sparse_graph():
  '''Graph from the slides referenced at the top.'''
  ii = np.array([0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 6, 7, 7, 7, 8])
  jj = np.array([1, 2, 0, 3, 0, 4, 5, 1, 6, 7, 8, 2, 7, 2, 7, 3, 3, 4, 5, 3])
  G = np.zeros((9,9))
  G[ii,jj] = 1
  return G


if __name__ == '__main__':
  main_sparse(test_sparse_graph())
