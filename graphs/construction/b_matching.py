from __future__ import absolute_import, print_function
import numpy as np
import warnings
from graphs import Graph
from ..mini_six import range

__all__ = ['b_matching']


def b_matching(D, k, max_iter=1000, damping=1, conv_thresh=1e-4,
               verbose=False):
  '''
  "Belief-Propagation for Weighted b-Matchings on Arbitrary Graphs
  and its Relation to Linear Programs with Integer Solutions"
  Bayati et al.

  Finds the minimal weight perfect b-matching using min-sum loopy-BP.

  @param D pairwise distance matrix
  @param k number of neighbors per vertex (scalar or array-like)

  Based on the code at http://www.cs.columbia.edu/~bert/code/bmatching/bdmatch
  '''
  INTERVAL = 2
  oscillation = 10
  cbuff = np.zeros(100, dtype=float)
  cbuffpos = 0
  N = D.shape[0]
  assert D.shape[1] == N, 'Input distance matrix must be square'
  mask = ~np.eye(N, dtype=bool)  # Assume all nonzero except for diagonal
  W = -D[mask].reshape((N, -1)).astype(float)
  degrees = np.clip(np.atleast_1d(k), 0, N-1)
  if degrees.size == 1:  # broadcast scalar up to length-N array
    degrees = np.repeat(degrees, N)
  else:
    assert degrees.shape == (N,), 'Input degrees must have length N'
  # TODO: remove these later
  inds = np.tile(np.arange(N), (N, 1))
  backinds = inds.copy()
  inds = inds[mask].reshape((N, -1))
  backinds = backinds.T.ravel()[:(N*(N-1))].reshape((N, -1))

  # Run Belief Revision
  change = 1.0
  B = W.copy()
  for n_iter in range(1, max_iter+1):
    oldB = B.copy()
    update_belief(oldB, B, W, degrees, damping, inds, backinds)

    # check for convergence
    if n_iter % INTERVAL == 0:
      # track changes
      c = np.abs(B[:,0]).sum()
      # c may be infinite here, and that's ok
      with np.errstate(invalid='ignore'):
        if np.any(np.abs(c - cbuff) < conv_thresh):
          oscillation -= 1
      cbuff[cbuffpos] = c
      cbuffpos = (cbuffpos + 1) % len(cbuff)

      change = diff_belief(B, oldB)
      if np.isnan(change):
        warnings.warn("change is NaN! "
                      "BP will quit but solution could be invalid. "
                      "Problem may be infeasible.")
        break
      if change < conv_thresh or oscillation < 1:
        break
  else:
    warnings.warn("Hit iteration limit (%d) before converging" % max_iter)

  if verbose:  # pragma: no cover
    if change < conv_thresh:
      print("Converged to stable beliefs in %d iterations" % n_iter)
    elif oscillation < 1:
      print("Stopped after reaching oscillation in %d iterations" % n_iter)
      print("No feasible solution found or there are multiple maxima.")
      print("Outputting best approximate solution. Try damping.")

  # recover result from B
  thresholds = np.zeros(N)
  for i,d in enumerate(degrees):
    Brow = B[i]
    if d >= N - 1:
      thresholds[i] = -np.inf
    elif d < 1:
      thresholds[i] = np.inf
    else:
      thresholds[i] = Brow[quickselect(-Brow, d-1)]

  ii,jj = np.where(B >= thresholds[:,None])
  pairs = np.column_stack((ii, inds[ii,jj]))
  return Graph.from_edge_pairs(pairs, num_vertices=N)


def _update_change(B, oldB):  # pragma: no cover
  expB = np.exp(B)
  expB[np.isinf(expB)] = 0
  rowsums = expB.sum(axis=1)
  expOldB = np.exp(oldB)
  expOldB[np.isinf(expOldB)] = 0
  oldrowsums = expOldB.sum(axis=1)

  change = 0
  rowsums[rowsums==0] = 1
  oldrowsums[oldrowsums==0] = 1
  for i in range(B.shape[0]):
    row = expB[i]
    oldrow = expOldB[i]
    rmask = row == 0
    ormask = oldrow == 0
    change += np.count_nonzero(np.logical_xor(rmask, ormask))
    mask = ~np.logical_and(rmask, ormask)
    change += np.abs(oldrow[mask]/oldrowsums[i] -
                     row[mask]/rowsums[i]).sum()
  return change


def _quickselect(B_row, *ks):  # pragma: no cover
  order = np.argpartition(B_row, ks)
  if len(ks) == 1:
    return order[ks[0]]
  return [order[k] for k in ks]


def _updateB(oldB, B, W, degrees, damping, inds, backinds):  # pragma: no cover
  '''belief update function.'''
  for j,d in enumerate(degrees):
    kk = inds[j]
    bk = backinds[j]

    if d == 0:
      B[kk,bk] = -np.inf
      continue

    belief = W[kk,bk] + W[j]
    oldBj = oldB[j]
    if d == oldBj.shape[0]:
      bth = quickselect(-oldBj, d-1)
      bplus = -1
    else:
      bth,bplus = quickselect(-oldBj, d-1, d)

    belief -= np.where(oldBj >= oldBj[bth], oldBj[bplus], oldBj[bth])
    B[kk,bk] = damping*belief + (1-damping)*oldB[kk,bk]


try:
  import pyximport
  pyximport.install(setup_args={'include_dirs': np.get_include()})
  from ._fast_paths import quickselect, update_belief, diff_belief
except ImportError:
  quickselect = _quickselect
  update_belief = _updateB
  diff_belief = _update_change
