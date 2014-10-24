# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True
import numpy as np
cimport numpy as np
cimport cython
from libcpp cimport bool
from common.distance import SquaredL2

IDX_DTYPE = np.intp
ctypedef Py_ssize_t IDX_DTYPE_t


def inter_cluster_distance(X, num_clusters, cluster_labels, metric=SquaredL2):
  # compute shortest distances between clusters
  Dx = metric.within(X)
  Dc = np.zeros((num_clusters,num_clusters), dtype=np.float64)
  edges = np.zeros((num_clusters,num_clusters,2), dtype=IDX_DTYPE)
  _fill_Dc_edges(num_clusters, cluster_labels, Dx, Dc, edges)
  return Dc, edges


cdef void _fill_Dc_edges(IDX_DTYPE_t num_clusters,
                         int[::1] cluster_labels,
                         double[:,::1] Dx,
                         double[:,::1] Dc,
                         IDX_DTYPE_t[:,:,::1] edges):
  cdef IDX_DTYPE_t i, j, k, l, r, c, ik, il, ii_n, jj_n
  cdef double min_val
  cdef double INF = np.inf
  cdef IDX_DTYPE_t n = Dx.shape[0]
  cdef bool[:,::1] masks
  cdef IDX_DTYPE_t[::1] ii, jj
  cdef list indices = []
  for i in range(num_clusters):
    indices.append(where_eq(cluster_labels, i))
  for i in range(num_clusters-1):
    ii = indices[i]
    ii_n = ii.shape[0]
    for j in range(i+1, num_clusters):
      jj = indices[j]
      jj_n = jj.shape[0]
      r = 0
      c = 0
      min_val = INF
      for ik in range(ii_n):
        k = ii[ik]
        for il in range(jj_n):
          l = jj[il]
          if Dx[k,l] < min_val:
            min_val = Dx[k,l]
            # Transposed index
            r = k
            c = l
      edges[i,j,0] = r
      edges[i,j,1] = c
      edges[j,i,0] = r
      edges[j,i,1] = c
      Dc[i,j] = min_val
      Dc[j,i] = min_val

cdef IDX_DTYPE_t[::1] where_eq(int[::1] x, IDX_DTYPE_t val):
  # return np.where(x == val)[0]
  cdef IDX_DTYPE_t n = x.shape[0]
  cdef IDX_DTYPE_t i, n_inds
  cdef list inds = []
  for i in range(n):
    if x[i] == val:
      inds.append(i)
  n_inds = len(inds)
  cdef IDX_DTYPE_t[::1] result = np.empty(n_inds, dtype=IDX_DTYPE)
  for i in range(n_inds):
    result[i] = inds[i]
  return result
