import matplotlib
matplotlib.use('template')

import unittest
import numpy as np
import warnings
from numpy.testing import assert_array_equal, assert_array_almost_equal
from scipy.sparse import coo_matrix
from graphs import analysis, Graph

PAIRS = np.array([[0,1],[0,2],[1,2],[2,0],[3,4],[4,3]])
ADJ = [[0,1,1,0,0],
       [0,0,1,0,0],
       [1,0,0,0,0],
       [0,0,0,0,1],
       [0,0,0,1,0]]


class TestAnalysis(unittest.TestCase):
  def setUp(self):
    self.graphs = [
        Graph.from_edge_pairs(PAIRS),
        Graph.from_adj_matrix(ADJ),
        Graph.from_adj_matrix(coo_matrix(ADJ)),
    ]

  def test_connected_components(self):
    for G in self.graphs:
      n, labels = analysis.connected_components(G)
      self.assertEqual(2, n)
      assert_array_equal(labels, [0,0,0,1,1])

  def test_greedy_coloring(self):
    for G in self.graphs:
      assert_array_equal([1,2,3,1,2], analysis.greedy_coloring(G.symmetrize()))

  def test_ave_laplacian(self):
    g = Graph.from_adj_matrix([[0,1,2],[1,0,0],[2,0,0]])
    expected = np.array([[1,-0.5,0],[-0.5,1,0],[0,0,1]])
    assert_array_almost_equal(analysis.ave_laplacian(g), expected)

  def test_directed_laplacian(self):
    expected = np.array([
        [0.239519, -0.05988, -0.179839, 0,   0],
        [-0.05988,  0.120562,-0.060281, 0,   0],
        [-0.179839,-0.060281, 0.239919, 0,   0],
        [0,         0,        0,        0.2,-0.2],
        [0,         0,        0,       -0.2, 0.2]])
    for G in self.graphs:
      L = analysis.directed_laplacian(G)
      assert_array_almost_equal(L, expected)

    # test non-convergence case
    with warnings.catch_warnings(record=True) as w:
      analysis.directed_laplacian(self.graphs[0], max_iter=2)
      self.assertEqual(len(w), 1)
      self.assertEqual(w[0].message.message,
                       'phi failed to converge after 2 iterations')

  def test_bottlenecks(self):
    for G in self.graphs:
      b = analysis.bottlenecks(G)
      assert_array_equal(b, [[0, 1]])
    for G in self.graphs:
      b,c = analysis.bottlenecks(G, n=2, return_counts=True)
      assert_array_equal(b, [[0, 1], [1, 2]])
      assert_array_equal(c, [1, 1])
    for G in self.graphs:
      b = analysis.bottlenecks(G, directed=True)
      assert_array_equal(b, [[2, 0]])

  def test_bandwidth(self):
    for G in self.graphs:
      self.assertEqual(analysis.bandwidth(G), 2)

  def test_profile(self):
    for G in self.graphs:
      self.assertEqual(analysis.profile(G), 1)


if __name__ == '__main__':
  unittest.main()
