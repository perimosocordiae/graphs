import matplotlib
matplotlib.use('template')

import numpy as np
import unittest
from graphs import Graph, reorder


class TestReorder(unittest.TestCase):
  def setUp(self):
    ii = np.array([0, 0, 1, 2, 2, 3, 3, 3, 4, 5])
    jj = np.array([1, 2, 3, 4, 5, 6, 7, 8, 7, 7])
    adj = np.zeros((9,9), dtype=int)
    adj[ii,jj] = 1
    adj[jj,ii] = 1
    self.G = Graph.from_adj_matrix(adj)

  def test_cuthill_mckee(self):
    # Many orderings are "correct". Just ensure minimal bandwidth.
    expected_b = 3
    # test default version (probably scipy)
    cm = reorder.cuthill_mckee(self.G)
    self.assertEqual(reorder.bandwidth(cm), expected_b)
    # test the non-scipy version
    cm = reorder._cuthill_mckee(self.G)
    self.assertEqual(reorder.bandwidth(cm), expected_b)

  @unittest.expectedFailure
  def test_node_centroid_hill_climbing(self):
    np.random.seed(1234)
    nchc = reorder.node_centroid_hill_climbing(self.G)
    # TODO: debug
    self.assertEqual(reorder.bandwidth(nchc), 3)

  @unittest.expectedFailure
  def test_laplacian_reordering(self):
    lap = reorder.laplacian_reordering(self.G)
    # TODO: debug
    self.assertEqual(reorder.bandwidth(lap), 3)


if __name__ == '__main__':
  unittest.main()
