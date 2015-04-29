import numpy as np
import unittest

from graphs.generators.shapes import SCurve
from graphs.construction import directed_graph


class TestDirected(unittest.TestCase):

  def test_directed_graph(self):
    # XXX: This kind of testing isn't exactly reproducible across versions
    np.random.seed(1234)
    traj = SCurve().trajectories(5, 20)
    G, X = directed_graph(traj, k=5, pruning_thresh=0, return_coords=True)
    P = directed_graph(traj, k=5, pruning_thresh=0.1)
    self.assertEqual(X.shape, (100, 3))
    self.assertEqual(G.num_edges(), 500)
    # Results may vary slightly, so just check that we're <500
    self.assertLess(P.num_edges(), 500)


if __name__ == '__main__':
  unittest.main()
