# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True
import numpy as np
cimport numpy as np
cimport cython

IDX_DTYPE = np.intp
ctypedef Py_ssize_t IDX_DTYPE_t

cdef extern from "math.h":
    float INFINITY

def update_change(B, oldB):
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


def quickselect(B_row, IDX_DTYPE_t k):
  cdef IDX_DTYPE_t[::1] order = np.argpartition(-B_row, k)
  return order[k]


def updateB(oldB, double[:,::1] B, double[:,::1] W,
            IDX_DTYPE_t[::1] degrees, double damping,
            IDX_DTYPE_t[:,::1] inds, IDX_DTYPE_t[:,::1] backinds):
  '''belief update function.'''
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
    # TODO: handle case with degree < 1
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
