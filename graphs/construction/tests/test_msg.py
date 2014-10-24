import matplotlib
matplotlib.use('template')

import numpy as np
import unittest

from graphs.generators.swiss_roll import swiss_roll, error_ratio
from graphs.construction import manifold_spanning_graph


class TestMSG(unittest.TestCase):

  def test_manifold_spanning_graph(self):
    np.random.seed(1234)
    X, theta = swiss_roll(6, 120, radius=4.8, return_theta=True)
    GT = np.hstack((theta[:,None], X[:,1:2]))
    GT -= GT.min(axis=0)
    GT /= GT.max(axis=0)

    G = manifold_spanning_graph(X, 2)
    self.assertEqual(error_ratio(G, GT), 0.0)


if __name__ == '__main__':
  unittest.main()
