from __future__ import absolute_import

import numpy as np
import warnings
from sklearn.metrics.pairwise import pairwise_distances

from ..mini_six import range
from .neighbors import nearest_neighbors

__all__ = [
    'downsample_trajectories', 'epsilon_net', 'fuzzy_c_means'
]


def downsample_trajectories(trajectories, downsampler, *args, **kwargs):
  '''Downsamples all points together, then re-splits into original trajectories.

  trajectories : list of 2-d arrays, each representing a trajectory
  downsampler(X, *args, **kwargs) : callable that returns indices into X
  '''
  X = np.vstack(trajectories)
  traj_lengths = list(map(len, trajectories))
  inds = np.sort(downsampler(X, *args, **kwargs))
  new_traj = []
  for stop in np.cumsum(traj_lengths):
    n = np.searchsorted(inds, stop)
    new_traj.append(X[inds[:n]])
    inds = inds[n:]
  return new_traj


def epsilon_net(points, close_distance):
  '''Selects a subset of `points` to preserve graph structure while minimizing
  the number of points used, by removing points within `close_distance`.
  Returns the downsampled indices.'''
  num_points = points.shape[0]
  indices = set(range(num_points))
  selected = []
  while indices:
    idx = indices.pop()
    nn_inds, = nearest_neighbors(points[idx], points, epsilon=close_distance)
    indices.difference_update(nn_inds)
    selected.append(idx)
  return selected


def fuzzy_c_means(points, num_centers, m=2., tol=1e-4, max_iter=100,
                  verbose=False):
  '''Uses Fuzzy C-Means to downsample `points`.
  m : aggregation parameter >1, larger implies smoother clusters
  Returns indices of downsampled points.
  '''
  num_points = points.shape[0]
  if num_centers >= num_points:
    return np.arange(num_points)
  # randomly initialize cluster assignments matrix
  assn = np.random.random((points.shape[0], num_centers))
  # iterate assignments until they converge
  for i in range(max_iter):
    # compute centers
    w = assn ** m
    w /= w.sum(axis=0)
    centers = w.T.dot(points)
    # calculate new assignments
    d = pairwise_distances(points, centers)
    d **= 2. / (m - 1)
    np.maximum(d, 1e-10, out=d)
    new_assn = 1. / np.einsum('ik,ij->ik', d, 1./d)
    # check for convergence
    change = np.linalg.norm(new_assn - assn)
    if verbose:
      print('At iteration %d: change = %g' % (i+1, change))
    if change < tol:
      break
    assn = new_assn
  else:
    warnings.warn("fuzzy_c_means didn't converge in %d iterations" % max_iter)
  # find points closest to the selected cluster centers
  return d.argmin(axis=0)
