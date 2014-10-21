import matplotlib
matplotlib.use('template')

import unittest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from scipy.sparse import coo_matrix
from graphs import (
    analysis, EdgePairGraph, DenseAdjacencyMatrixGraph,
    SparseAdjacencyMatrixGraph
)

PAIRS = np.array([[0,1],[0,2],[1,2],[2,0],[3,4],[4,3]])
ADJ = [[0,1,1,0,0],
       [0,0,1,0,0],
       [1,0,0,0,0],
       [0,0,0,0,1],
       [0,0,0,1,0]]


class TestAnalysis(unittest.TestCase):
  def setUp(self):
    self.graphs = [
        EdgePairGraph(PAIRS),
        DenseAdjacencyMatrixGraph(ADJ),
        SparseAdjacencyMatrixGraph(coo_matrix(ADJ)),
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
    g = DenseAdjacencyMatrixGraph([[0,1,2],[1,0,0],[2,0,0]])
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

  def test_bottlenecks(self):
    for G in self.graphs:
      b = analysis.bottlenecks(G)
      assert_array_equal(b, [[0,1]])
    for G in self.graphs:
      b = analysis.bottlenecks(G, directed=True)
      assert_array_equal(b, [[4,3]])


if __name__ == '__main__':
  unittest.main()
