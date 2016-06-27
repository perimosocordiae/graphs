# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True
import numpy as np
cimport numpy as np
cimport cython
from libcpp cimport bool
from sklearn.metrics import pairwise_distances

cdef extern from "math.h":
    float INFINITY

IDX_DTYPE = np.intp
ctypedef Py_ssize_t IDX_DTYPE_t


def find_relative_neighbors(D):
  cdef IDX_DTYPE_t n = D.shape[0]
  cdef IDX_DTYPE_t max_num_pairs = n * (n-1) // 2
  pairs = np.empty((max_num_pairs, 2), dtype=IDX_DTYPE)
  cdef IDX_DTYPE_t end_idx = _fill_rn_pairs(D, n, pairs)
  return pairs[:end_idx]


cdef IDX_DTYPE_t _fill_rn_pairs(double[:,::1] D,
                                IDX_DTYPE_t n,
                                IDX_DTYPE_t[:,::1] pairs):
  cdef IDX_DTYPE_t idx = 0
  cdef IDX_DTYPE_t r, c, i
  cdef double d
  for r in range(n-1):
    for c in range(r+1, n):
      d = D[r,c]
      for i in range(n):
        if i == r or i == c:
          continue
        if D[r,i] < d and D[c,i] < d:
          break  # Point in lune, this is not an edge
      else:
        pairs[idx,0] = r
        pairs[idx,1] = c
        idx += 1
  return idx


def inter_cluster_distance(X, num_clusters, cluster_labels):
  # compute shortest distances between clusters
  Dx = pairwise_distances(X, metric='sqeuclidean')
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


cpdef IDX_DTYPE_t quickselect(B_row, IDX_DTYPE_t k):
  cdef IDX_DTYPE_t[::1] order = np.argpartition(B_row, k)
  return order[k]


def diff_belief(B, oldB):
  cdef IDX_DTYPE_t i, j, N = B.shape[0]
  cdef double rs, ors, change = 0
  cdef double[::1] rowsums, oldrowsums, row, oldrow
  expB = np.exp(B)
  expOldB = np.exp(oldB)
  expB[np.isinf(expB)] = 0
  expOldB[np.isinf(expOldB)] = 0
  rowsums = expB.sum(axis=1)
  oldrowsums = expOldB.sum(axis=1)

  for i in range(N):
    rs = rowsums[i]
    ors = oldrowsums[i]
    if rs == 0:
      rs = 1
    if ors == 0:
      ors = 1
    row = expB[i]
    oldrow = expOldB[i]
    for j in range(N-1):
      if (row[j] == 0 and oldrow[j] != 0) or (row[j] != 0 and oldrow[j] == 0):
        change += 1
      if row[j] != 0 and oldrow[j] != 0:
        change += abs(oldrow[j]/ors - row[j]/rs)
  return change


def update_belief(oldB, double[:,::1] B, double[:,::1] W,
            IDX_DTYPE_t[::1] degrees, double damping,
            IDX_DTYPE_t[:,::1] inds, IDX_DTYPE_t[:,::1] backinds):
  cdef IDX_DTYPE_t j, d, kkk, bkk, n = degrees.shape[0]
  cdef IDX_DTYPE_t[::1] kk, bk, order
  cdef IDX_DTYPE_t bth, bplus
  cdef double[::1] oldBj
  cdef double[:,::1] oldBview = oldB
  cdef double belief
  for j in range(n):
    d = degrees[j]
    kk = inds[j]
    bk = backinds[j]

    if d == 0:
      for k in range(n-1):
        kkk = kk[k]
        bkk = bk[k]
        B[kkk,bkk] = -INFINITY
      continue

    oldBj = oldBview[j]
    if d == n-1:
      bth = quickselect(-oldB[j], d-1)
      bplus = -1
    else:
      order = np.argpartition(-oldB[j], (d-1, d))
      bth = order[d-1]
      bplus = order[d]

    for k in range(n-1):
      kkk = kk[k]
      bkk = bk[k]
      belief = W[kkk,bkk] + W[j,k]

      if oldBj[k] >= oldBj[bth]:
        belief -= oldBj[bplus]
      else:
        belief -= oldBj[bth]
      B[kkk,bkk] = damping*belief + (1-damping)*oldBview[kkk,bkk]
