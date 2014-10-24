import matplotlib
matplotlib.use('template')

import numpy as np
import unittest
from numpy.testing import assert_array_equal

from graphs.generators.shapes import SCurve
from graphs.construction import jeff_graph, jeff_prune_edges


class TestDirected(unittest.TestCase):

  def test_jeff_graph(self):
    np.random.seed(1234)
    traj = SCurve().trajectories(5, 20)
    G, X = jeff_graph(traj, k=5, pruning_thresh=0, return_coords=True)
    P = jeff_graph(traj, k=5, pruning_thresh=0.1)
    self.assertEqual(X.shape, (100, 3))
    self.assertEqual(G.num_edges(), 500)
    self.assertEqual(P.num_edges(), 419)


if __name__ == '__main__':
  unittest.main()
