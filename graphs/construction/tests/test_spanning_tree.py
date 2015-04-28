import numpy as np
import unittest
from numpy.testing import assert_array_almost_equal
from sklearn.metrics import pairwise_distances

from graphs.construction import mst, perturbed_mst, disjoint_mst


class TestSpanningTree(unittest.TestCase):
  def setUp(self):
    self.pts = np.array([[0,0],[1,2],[3,2],[-1,0]])

  def test_mst(self):
    expected = [[0,    2.236,0, 1],
                [2.236,0,    2, 0],
                [0,    2,    0, 0],
                [1,    0,    0, 0]]
    G = mst(self.pts)
    assert_array_almost_equal(G.matrix(dense=True), expected, decimal=3)
    # Check precomputed metric.
    D = pairwise_distances(self.pts)
    G = mst(D, metric='precomputed')
    assert_array_almost_equal(G.matrix(dense=True), expected, decimal=3)

  def test_perturbed_mst(self):
    np.random.seed(1234)
    expected = [[0,0.71428571,0.23809524,1.00000000],
                [0.71428571,0,0.85714286,0.14285714],
                [0.23809524,0.85714286,0,0.04761905],
                [1.00000000,0.14285714,0.04761905,0]]
    G = perturbed_mst(self.pts)
    assert_array_almost_equal(G.matrix(dense=True), expected)

  def test_disjoint_mst(self):
    expected = [[0,2.23606798,3.60555128,1],
                [2.23606798,0,2,2.82842712],
                [3.60555128,2,0,4.47213595],
                [1,2.82842712,4.47213595,0]]
    G = disjoint_mst(self.pts)
    assert_array_almost_equal(G.matrix(dense=True), expected)

if __name__ == '__main__':
  unittest.main()
