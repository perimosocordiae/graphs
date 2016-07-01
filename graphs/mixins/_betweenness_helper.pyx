#distutils: language = c++
#cython: boundscheck=False, wraparound=True, cdivision=True
cimport numpy as np
import numpy as np
import scipy.sparse as ss
from libcpp.deque cimport deque
from libcpp.stack cimport stack
from libcpp.pair cimport pair
from libcpp.queue cimport priority_queue

cdef double INF = float('inf')
ctypedef np.int_t intc
ctypedef dict (*sssp_fn)(object, intc, intc[::1], double[::1], stack[intc]&)

cpdef betweenness(adj, bint weighted, bint vertex):
  cdef sssp_fn sssp
  if weighted:
    sssp = &_sssp_weighted
  else:
    sssp = &_sssp_unweighted
  # sigma[v]: number of shortest paths from s->v
  # delta[v]: dependency of s on v
  cdef intc s, w, v, n = adj.shape[0]
  cdef double[::1] delta = np.zeros(n)
  cdef double[::1] dist = np.zeros(n)
  cdef intc[::1] sigma = np.zeros(n, dtype=np.int)
  cdef double coeff
  cdef stack[intc] S
  cdef dict pred
  cdef double[::1] vbtw
  if vertex:
    # Brandes algorithm for vertex betweenness
    vbtw = np.zeros(n)
    for s in range(n):
      pred = sssp(adj, s, sigma, dist, S)
      delta[:] = 0
      while not S.empty():
        w = S.top()
        coeff = (1.0 + delta[w]) / sigma[w]
        for v in pred.get(w, []):
          delta[v] += sigma[v] * coeff
        if w != s:
          vbtw[w] += delta[w]
        S.pop()
    return np.array(vbtw, dtype=float)
  # Brandes variant for edge betweennes
  # set up betweenness container with correct sparsity pattern
  ebtw = ss.csr_matrix(adj, dtype=float, copy=True)
  ebtw.eliminate_zeros()
  ebtw.data[:] = 0
  for s in range(n):
    pred = sssp(adj, s, sigma, dist, S)
    delta[:] = 0
    while not S.empty():
      w = S.top()
      coeff = (1.0 + delta[w]) / sigma[w]
      for v in pred.get(w, []):
        c = sigma[v] * coeff
        ebtw[v,w] += c
        delta[v] += c
      S.pop()
  return ebtw.data


cdef dict _sssp_unweighted(adj, intc s, intc[::1] sigma, double[::1] dist, stack[intc]& S):
  cdef intc v, w, i, j, widx
  cdef double new_weight
  cdef dict pred = {}
  sigma[:] = 0
  sigma[s] = 1
  dist[:] = INF
  dist[s] = 0
  cdef deque[intc] Q
  Q.push_back(s)
  while not Q.empty():
    v = Q.front()
    Q.pop_front()
    S.push(v)
    new_weight = dist[v] + 1
    i = adj.indptr[v]
    j = adj.indptr[v+1]
    for widx in range(i, j):
      w = adj.indices[widx]
      if dist[w] > new_weight:
        pred[w] = [v]
        sigma[w] = sigma[v]
        dist[w] = new_weight
        Q.push_back(w)
      elif dist[w] == new_weight:
        pred[w].append(v)
        sigma[w] += sigma[v]
  return pred


cdef dict _sssp_weighted(adj, intc s, intc[::1] sigma, double[::1] dist, stack[intc]& S):
  cdef intc v, w, i, j, widx
  cdef double dist_v, new_weight, d
  cdef set SS = set()
  cdef dict pred = {}
  sigma[:] = 0
  sigma[s] = 1
  dist[:] = INF
  dist[s] = 0
  cdef priority_queue[pair[double,intc]] Q
  Q.push(pair[double,intc](0.,s))
  while not Q.empty():
    tmp = Q.top()
    Q.pop()
    dist_v = tmp.first
    v = tmp.second
    SS.add(v)
    i = adj.indptr[v]
    j = adj.indptr[v+1]
    for widx in range(i, j):
      w = adj.indices[widx]
      d = adj.data[widx]
      new_weight = dist_v + d
      if dist[w] > new_weight:
        pred[w] = [v]
        sigma[w] = sigma[v]
        dist[w] = new_weight
        Q.push(pair[double,intc](new_weight, w))
      elif dist[w] == new_weight:
        pred[w].append(v)
        sigma[w] += sigma[v]
  # XXX: ugly workaround: using lambdas/comprehensions in cdef -> segfault
  cdef list foo = [(dist[v], v) for v in SS]
  for _,w in sorted(foo):
    S.push(w)
  return pred
