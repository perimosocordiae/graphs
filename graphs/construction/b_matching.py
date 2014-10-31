import numpy as np
from graphs import Graph

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
  degrees = np.zeros(N, dtype=int) + k
  # TODO: remove these later
  inds = np.tile(np.arange(N), (N, 1))
  backinds = inds.copy()
  inds = inds[mask].reshape((N, -1))
  backinds = backinds.T.ravel()[:(N*(N-1))].reshape((N, -1))

  # Run Belief Revision
  change = 1.0
  B = W.copy()
  for n_iter in xrange(1, max_iter+1):
    oldB = B.copy()
    updateB(oldB, B, W, degrees, damping, inds, backinds)

    # check for convergence
    if n_iter % INTERVAL == 0:
      # track changes
      cbuff[cbuffpos] = 0
      for i in xrange(N):
        cbuff[cbuffpos] += abs(B[i,0])

      for i,c in enumerate(cbuff):
        if i != cbuffpos and abs(c-cbuff[cbuffpos]) < conv_thresh:
          oscillation -= 1
          break

      cbuffpos += 1
      if cbuffpos >= len(cbuff):
        cbuffpos = 0

      expB = np.exp(B)
      expB[np.isinf(expB)] = 0
      rowsums = expB.sum(axis=1)
      expOldB = np.exp(oldB)
      expOldB[np.isinf(expOldB)] = 0
      oldrowsums = expOldB.sum(axis=1)

      change = 0
      for i in xrange(N):
        if rowsums[i] == 0:
          rowsums[i] = 1
        if oldrowsums[i] == 0:
          oldrowsums[i] = 1
        for j in xrange(N-1):
          if ((expOldB[i,j] == 0 and expB[i,j] != 0) or
              (expOldB[i,j] != 0 and expB[i,j] == 0)):
            change += 1
          elif expOldB[i,j] == 0 and expB[i,j] == 0:
            change = change
          else:
            change += abs(expOldB[i,j]/oldrowsums[i]- expB[i,j]/rowsums[i])
      if np.isnan(change):
        print "change is NaN! BP will quit but solution",
        print "could be invalid. Problem may be infeasible."
        break
      if change < conv_thresh or oscillation < 1:
        break
  else:
    print "Hit iteration limit (%d) before converging" % max_iter

  if verbose:
    if change < conv_thresh:
      print "Converged to stable beliefs in %d iterations" % n_iter
    elif oscillation < 1:
      print "Stopped after reaching oscillation in %d iterations" % n_iter
      print "No feasible solution found or there are multiple maxima."
      print "Outputting best approximate solution. Try damping."

  # recover result from B
  thresholds = np.zeros(N)
  for i,k in enumerate(degrees):
    poscount = np.count_nonzero(B[i] > 0)
    if degrees[i] >= N - 1:
      thresholds[i] = -np.inf
    elif k < 1:
      thresholds[i] = np.inf
    elif poscount >= k:
      thresholds[i] = B[i][quickselect(B[i],k-1)]
    elif poscount <= k:
      if k == 0:
        thresholds[i] = np.inf
      else:
        thresholds[i] = B[i][quickselect(B[i],k-1)]
    else:
        thresholds[i] = B[i][quickselect(B[i],poscount-1)]

  ii,jj = np.where(B >= thresholds[:,None])
  adj = np.zeros_like(D, dtype=bool)
  adj[ii,inds[ii,jj]] = True
  return Graph.from_adj_matrix(adj)


def quickselect(B_row, *ks):
  order = np.argpartition(-B_row, ks)
  if len(ks) == 1:
    return order[ks[0]]
  return [order[k] for k in ks]


def updateB(oldB, B, W, degrees, damping, inds, backinds):
  '''belief update function.'''
  N = len(degrees)
  for j in xrange(N):
    poscount = np.count_nonzero(B[j] > 0)

    if degrees[j] == 0:
      for i in xrange(N-1):
        k = inds[j,i]
        B[k,backinds[j,i]] = -np.inf
    elif ((degrees[j]<poscount and poscount<degrees[j]) or (degrees[j]==0 and poscount==0)):
      for i in xrange(N-1):
        k = inds[j,i]
        B[k,backinds[j,i]] = W[k,backinds[j,i]] + W[j,i]
    else:
      bth,bplus = quickselect(oldB[j], degrees[j]-1, degrees[j])

      for i in xrange(N-1):
        k = inds[j,i]
        bkji = backinds[j,i]

        if poscount <= degrees[j]:
          if oldB[j,i] >= oldB[j,bth]:
            if bplus < 0:
              B[k,bkji] = np.inf
            else:
              B[k,bkji] = W[k,bkji] + W[j,i] - oldB[j,bplus]
          else:
            B[k,bkji] = W[k,bkji] + W[j,i] - oldB[j,bth]
        elif poscount == degrees[j]:
          if oldB[j,i] >= oldB[j,bth]:
            B[k,bkji] = W[k,bkji] + W[j,i]
          else:
            B[k,bkji] = W[k,bkji] + W[j,i] - oldB[j,bth]
        elif poscount > degrees[j]:
          if oldB[j,i] >= oldB[j,bth]:
            if bplus < 0:
              B[k,bkji] = np.inf
            else:
              B[k,bkji] = W[k,bkji] + W[j,i] - oldB[j,bplus]
          else:
              B[k,bkji] = W[k,bkji] + W[j,i] - oldB[j,bth]

        B[k,bkji] = damping*B[k,bkji] + (1-damping)*oldB[k,bkji]
