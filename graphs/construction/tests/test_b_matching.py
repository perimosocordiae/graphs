import matplotlib
matplotlib.use('template')

import numpy as np
import unittest
from numpy.testing import assert_array_equal
from sklearn.metrics.pairwise import pairwise_distances

from graphs.construction import b_matching, b_matching_jebara


class TestBMatching(unittest.TestCase):
  def setUp(self):
    pts = np.array([[0,1,],[1,1],[0.2,0],[1,0],[3.4,3],[3.5,4.2]])
    self.dists = pairwise_distances(pts)
    self.expected = [
        [1,1,0,0,0,0],
        [1,1,0,0,0,0],
        [0,0,1,1,0,0],
        [0,0,1,1,0,0],
        [0,0,0,0,1,1],
        [0,0,0,0,1,1]
    ]

  def test_b_matching(self):
    G = b_matching(self.dists, 2)
    assert_array_equal(G.matrix(dense=True), self.expected)

  def test_b_matching_jebara(self):
    G = b_matching_jebara(self.dists, 2)
    assert_array_equal(G.matrix(dense=True), self.expected)


if __name__ == '__main__':
  unittest.main()
