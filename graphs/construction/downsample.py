import numpy as np
from neighbors import nearest_neighbors

__all__ = ['downsample', 'downsample_trajectories']


def downsample_trajectories(trajectories, close_distance):
  '''Downsamples all the points together,
     then re-splits into original trajectories.'''
  X = np.vstack(trajectories)
  traj_lengths = map(len, trajectories)
  inds = np.sort(downsample(X, close_distance))
  new_traj = []
  for stop in np.cumsum(traj_lengths):
    n = np.searchsorted(inds, stop)
    new_traj.append(X[inds[:n]])
    inds = inds[n:]
  return new_traj


def downsample(points, close_distance):
  '''Selects a subset of `points` to preserve graph structure while minimizing
  the number of points used, by removing points within `close_distance`.
  Returns the downsampled indices.'''
  num_points, num_dims = points.shape
  indices = set(range(num_points))
  selected = []
  while indices:
    idx = indices.pop()
    nn_inds, = nearest_neighbors(points[idx], points, epsilon=close_distance)
    indices.difference_update(nn_inds)
    selected.append(idx)
  return selected
