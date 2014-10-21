import numpy as np
from common.distance import SquaredL2
from common.neighborhood import nearest_neighbors


def downsample_trajectories(trajectories, close_distance, metric=SquaredL2):
  '''Downsamples all the points together,
     then re-splits into original trajectories.'''
  X = np.vstack(trajectories)
  traj_lengths = map(len, trajectories)
  inds = np.sort(downsample(X, close_distance, metric=metric))
  new_traj = []
  for stop in np.cumsum(traj_lengths):
    n = np.searchsorted(inds, stop)
    new_traj.append(X[inds[:n]])
    inds = inds[n:]
  return new_traj


def downsample(points, close_distance, metric=SquaredL2):
  '''Selects a subset of `points` to preserve graph structure while minimizing
  the number of points used, by removing points within `close_distance`.
  Returns the downsampled indices.'''
  num_points, num_dims = points.shape
  indices = set(range(num_points))
  selected = []
  while indices:
    idx = indices.pop()
    nn_inds, = nearest_neighbors(points[idx], points, metric=metric,
                                 epsilon=close_distance)
    indices.difference_update(nn_inds)
    selected.append(idx)
    #if not indices:
    #  selected.append(nn_inds[-1])
  return selected


if __name__ == '__main__':
  from common.synthetic_data import add_noise, spiral
  from common.viz import plot, pyplot
  n = 100000
  X = add_noise(spiral(np.random.random(n)*2*np.pi), 0.01)
  subset = downsample(X, 0.05)
  ax = pyplot.gca()
  plot(X, marker='x', ax=ax, label='Original Points')
  plot(X[subset], marker='o', ax=ax, label='Downsampled')
  pyplot.legend(loc='right')
  pyplot.show()
