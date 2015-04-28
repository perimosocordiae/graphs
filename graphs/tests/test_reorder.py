import numpy as np
import unittest
from numpy.testing import assert_array_equal
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
    self.assertEqual(cm.bandwidth(), expected_b)
    # test the non-scipy version
    cm = reorder._cuthill_mckee(self.G)
    self.assertEqual(cm.bandwidth(), expected_b)

  def test_node_centroid_hill_climbing(self):
    np.random.seed(1234)
    nchc = reorder.node_centroid_hill_climbing(self.G, relax=1)
    expected = np.array([[0,1],[0,2],[0,3],[0,4],[1,0],[2,0],[2,5],[3,0],[3,6],
                         [3,7],[4,0],[5,2],[5,8],[6,3],[6,8],[7,3],[7,8],[8,5],
                         [8,6],[8,7]])
    assert_array_equal(nchc.pairs(), expected)
    # test with relax < 1
    nchc2 = reorder.node_centroid_hill_climbing(self.G, relax=0.99)
    expected = np.array([[0,1],[1,0],[1,2],[1,3],[1,4],[2,1],[2,5],[3,1],[3,6],
                         [3,7],[4,1],[5,2],[5,8],[6,3],[6,8],[7,3],[7,8],[8,5],
                         [8,6],[8,7]])
    assert_array_equal(nchc2.pairs(), expected)

  def test_laplacian_reordering(self):
    lap = reorder.laplacian_reordering(self.G)
    self.assertEqual(lap.bandwidth(), 3)


if __name__ == '__main__':
  unittest.main()
