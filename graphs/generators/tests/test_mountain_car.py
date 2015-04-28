import numpy as np
import unittest
from matplotlib import pyplot
pyplot.switch_backend('template')

from graphs import Graph
from graphs.generators import mountain_car as mcar


class TestMountainCar(unittest.TestCase):

  def test_traj_sampling(self):
    traj, traces = mcar.sample_mcar_trajectories(3)
    self.assertEqual(len(traces), 3)
    self.assertEqual(len(traj), 3)
    self.assertEqual(traj[0].shape[1], 2)
    self.assertEqual(traj[1].shape[1], 2)
    self.assertEqual(traj[2].shape[1], 2)

  def test_basis_plotting(self):
    pts = np.random.random((5, 2))
    G = Graph.from_adj_matrix(np.random.random((5,5)))
    mcar.plot_mcar_basis(G, pts)


if __name__ == '__main__':
  unittest.main()
