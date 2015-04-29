import numpy as np
import unittest
import warnings
from numpy.testing import assert_array_equal
from sklearn.metrics.pairwise import pairwise_distances

from graphs.construction import b_matching


class TestBMatching(unittest.TestCase):
  def setUp(self):
    pts = np.array([
        [0.192,0.622],[0.438,0.785],[0.780,0.273],[0.276,0.802],[0.958,0.876],
        [0.358,0.501],[0.683,0.713],[0.370,0.561],[0.503,0.014],[0.773,0.883]])
    self.dists = pairwise_distances(pts)

  def test_standard(self):
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

  def test_warn_nonconvergence(self):
    with warnings.catch_warnings(record=True) as w:
      b_matching(self.dists, 2, max_iter=2)
      self.assertEqual(len(w), 1)
      self.assertEqual(str(w[0].message),
                       'Hit iteration limit (2) before converging')

  def test_oscillation(self):
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

  def test_array_b(self):
    b = np.zeros(10, dtype=int)
    b[5:] = 20
    expected = 1 - np.eye(10, dtype=int)
    expected[:5] = 0
    G = b_matching(self.dists, b)
    assert_array_equal(G.matrix(dense=True).astype(int), expected)

if __name__ == '__main__':
  unittest.main()
