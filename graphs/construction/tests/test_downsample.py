import numpy as np
import unittest
from numpy.testing import assert_array_equal

from graphs.generators.shapes import SCurve
from graphs.construction import downsample as d


class TestDownsample(unittest.TestCase):

  def test_epsilon_net(self):
    pts = np.array([[0,0],[1,2],[3,2],[-1,0]])
    sample = d.epsilon_net(pts, 1.7)
    self.assertTupleEqual(tuple(sample), (0,1,2))
    traj = [pts[:2], pts[2:]]
    sample = d.downsample_trajectories(traj, d.epsilon_net, 1.7)
    assert_array_equal(sample[0], pts[:2])
    assert_array_equal(sample[1], pts[2:3])

  def test_fuzzy_c_means(self):
    pts = np.array([[0,0],[1,2],[3,2],[-1,0]])
    sample = np.sort(d.fuzzy_c_means(pts, 2))
    assert_array_equal(sample, [0, 2])

  def test_downsample_trajectories(self):
    traj = SCurve().trajectories(5, 20)
    pts = np.vstack(traj)
    ds_traj = d.downsample_trajectories(traj, d.epsilon_net, 0.05)
    ds_pts = pts[d.epsilon_net(pts, 0.05)]
    assert_array_equal(np.vstack(ds_traj), ds_pts)


if __name__ == '__main__':
  unittest.main()
