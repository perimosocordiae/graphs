import matplotlib
matplotlib.use('template')

import numpy as np
import unittest

from graphs.construction import reorder


class TestReorder(unittest.TestCase):
  def setUp(self):
    ii = np.array([0, 0, 1, 2, 2, 3, 3, 3, 4, 5])
    jj = np.array([1, 2, 3, 4, 5, 6, 7, 8, 7, 7])
    self.G = np.zeros((9,9), dtype=int)
    self.G[ii,jj] = 1
    self.G[jj,ii] = 1

  def test_cuthill_mckee(self):
    # Many orderings are "correct". Just ensure minimal bandwidth.
    expected_b = 3
    # test default version (probably scipy)
    cm_order = reorder.cuthill_mckee(self.G)
    self.assertEqual(reorder.bandwidth(self.G, cm_order), expected_b)
    # test the non-scipy version
    cm_order = reorder._cuthill_mckee(self.G)
    self.assertEqual(reorder.bandwidth(self.G, cm_order), expected_b)

  @unittest.skip('TODO: test NCHC')
  def test_node_centroid_hill_climbing(self):
    np.random.seed(1234)
    nchc_order = reorder.node_centroid_hill_climbing(self.G)
    self.assertEqual(len(nchc_order), self.G.shape[0])
    # TODO: test this

  @unittest.skip('TODO: test laplacian_reordering')
  def test_laplacian_reordering(self):
    lap_order = reorder.laplacian_reordering(self.G)
    self.assertEqual(len(lap_order), self.G.shape[0])
    # TODO: test this


if __name__ == '__main__':
  unittest.main()
