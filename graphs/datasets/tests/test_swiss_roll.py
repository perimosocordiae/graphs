from __future__ import absolute_import
import numpy as np
import unittest

from ... import Graph
from ..swiss_roll import swiss_roll, error_ratio


class TestSwissRoll(unittest.TestCase):

  def test_swiss_roll(self):
    X = swiss_roll(6, 10)
    self.assertEqual(X.shape, (10, 3))
    X, theta = swiss_roll(3.0, 25, theta_noise=0, radius_noise=0,
                          return_theta=True)
    self.assertEqual(X.shape, (25, 3))
    self.assertEqual(theta.shape, (25,))
    self.assertAlmostEqual(theta.max(), 3.0)

  def test_error_ratio(self):
    adj = np.diag(np.ones(3), k=1)
    G = Graph.from_adj_matrix(adj + adj.T)
    GT = np.tile(np.linspace(0, 1, adj.shape[0])**2, (2,1)).T
    err_edges, tot_edges = error_ratio(G, GT, return_tuple=True)
    self.assertEqual(err_edges, 6)
    self.assertEqual(tot_edges, 6)
    self.assertEqual(error_ratio(G, GT, max_delta_theta=0.2), 4/6.)
    self.assertEqual(error_ratio(G, GT, max_delta_theta=0.5), 2/6.)
    self.assertEqual(error_ratio(G, GT, max_delta_theta=1), 0.0)


if __name__ == '__main__':
  unittest.main()
