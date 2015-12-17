from __future__ import division, absolute_import, print_function
from collections import deque
from heapq import heappush, heappop
import numpy as np
import scipy.sparse as ss
from ..mini_six import range


def _betweenness(adj, weighted, vertex):
  fn = _brandes if vertex else _brandes_edges
  return fn(adj, weighted)


def _brandes(adj, weighted):
  # Brandes algorithm for vertex betweenness
  # sigma[v]: number of shortest paths from s->v
  # delta[v]: dependency of s on v
  sssp = _sssp_weighted if weighted else _sssp_unweighted
  n = adj.shape[0]
  btw = np.zeros(n)
  for s in range(n):
    S, pred, sigma = sssp(adj, s)
    delta = np.zeros(n)
    while S:
      w = S.pop()
      coeff = (1 + delta[w]) / sigma[w]
      for v in pred.get(w, []):
        delta[v] += sigma[v] * coeff
      if w != s:
        btw[w] += delta[w]
  return btw


def _brandes_edges(adj, weighted):
  sssp = _sssp_weighted if weighted else _sssp_unweighted
  n = adj.shape[0]
  # set up betweenness container with correct sparsity pattern
  btw = ss.csr_matrix(adj, dtype=float, copy=True)
  btw.eliminate_zeros()
  btw.data[:] = 0
  for s in range(n):
    S, pred, sigma = sssp(adj, s)
    delta = np.zeros(n)
    while S:
      w = S.pop()
      coeff = (1 + delta[w]) / sigma[w]
      for v in pred.get(w, []):
        c = sigma[v] * coeff
        btw[v,w] += c
        delta[v] += c
  return btw.data


def _sssp_unweighted(adj, s):
  n = adj.shape[0]
  S = []
  pred = {}
  sigma = np.zeros(n)
  sigma[s] = 1
  dist = sigma + np.inf
  dist[s] = 0
  Q = deque([s])
  while Q:
    v = Q.popleft()
    S.append(v)
    new_weight = dist[v] + 1
    neighbors = adj[v].nonzero()[-1]
    for w in neighbors:
      if np.isinf(dist[w]):
        pred[w] = [v]
        sigma[w] = sigma[v]
        dist[w] = new_weight
        Q.append(w)
      elif dist[w] == new_weight:
        pred[w].append(v)
        sigma[w] += sigma[v]
  return S, pred, sigma


def _sssp_weighted(adj, s):
  n = adj.shape[0]
  S = set()
  pred = {}
  sigma = np.zeros(n)
  sigma[s] = 1
  dist = sigma + np.inf
  dist[s] = 0
  Q = [(0,s)]
  while Q:
    dist_v, v = heappop(Q)
    S.add(v)
    neighbors = adj[v].nonzero()[-1]
    for w in neighbors:
      new_weight = dist_v + adj[v,w]
      if dist[w] > new_weight:
        pred[w] = [v]
        sigma[w] = sigma[v]
        dist[w] = new_weight
        heappush(Q, (new_weight, w))
      elif dist[w] == new_weight:
        pred[w].append(v)
        sigma[w] += sigma[v]
  S = sorted(S, key=lambda v: dist[v])
  return S, pred, sigma

try:
  import pyximport
  pyximport.install(setup_args={'include_dirs': np.get_include()})
  from ._betweenness_helper import betweenness
except ImportError:
  betweenness = _betweenness
