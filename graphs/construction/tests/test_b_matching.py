import matplotlib
matplotlib.use('template')

import numpy as np
import unittest
from numpy.testing import assert_array_equal
from sklearn.metrics.pairwise import pairwise_distances

from graphs.construction import b_matching


class TestBMatching(unittest.TestCase):
  def setUp(self):
    pts = np.array([
        [0.192,0.622],[0.438,0.785],[0.780,0.273],[0.276,0.802],[0.958,0.876],
        [0.358,0.501],[0.683,0.713],[0.370,0.561],[0.503,0.014],[0.773,0.883]])
    self.dists = pairwise_distances(pts)

  def test_b_matching(self):
    # Generated with the bdmatch binary (b=2,damp=0.5)
    expected = np.array([
        [0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0]]).T
    G = b_matching(self.dists, 2, damping=0.5)
    assert_array_equal(G.matrix(dense=True).astype(int), expected)

  def test_b_matching_oscillation(self):
    # Generated with the bdmatch binary (b=2,damp=1)
    expected = np.array([
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0]])
    G = b_matching(self.dists, 2, damping=1)
    assert_array_equal(G.matrix(dense=True).astype(int), expected)

if __name__ == '__main__':
  unittest.main()
