import matplotlib
matplotlib.use('template')

import unittest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from scipy import sparse
from sklearn.metrics.pairwise import pairwise_distances

from graphs.construction import neighbors


def ngraph(*a, **k):
    return neighbors.neighbor_graph(*a,**k).matrix(dense=True)


class TestNeighbors(unittest.TestCase):
  def setUp(self):
    self.pts = np.array([[0,0],[1,2],[3,2],[-1,0]])

  def test_neighbor_graph(self):
    self.assertRaises(AssertionError, ngraph, self.pts)

  def test_binary_weighting(self):
    expected = np.array([[0,1,1,1],[1,0,1,1],[1,1,0,0],[1,1,0,0]])
    for kwargs in [dict(k=2, symmetrize=True),
                   dict(epsilon=3.61),
                   dict(k=2, epsilon=100)]:
      assert_array_equal(ngraph(self.pts, **kwargs), expected, str(kwargs))
    expected = np.array([[0,1,0,1],[1,0,1,0],[1,1,0,0],[1,1,0,0]])
    assert_array_equal(ngraph(self.pts, k=2, symmetrize=False), expected)

  def test_no_weighting_k(self):
    exp = np.sqrt([[0,5,0,1],[5,0,4,0],[13,4,0,0],[1,8,0,0]])
    kw = dict(k=2, weighting='none')
    assert_array_almost_equal(ngraph(self.pts, symmetrize=False, **kw), exp)
    assert_array_almost_equal(ngraph(self.pts, symmetrize=True, **kw),
                              (exp+exp.T)/2)

  def test_no_weighting_eps(self):
    exp = np.sqrt([[0,5,13,1],[5,0,4,8],[13,4,0,0],[1,8,0,0]])
    kw = dict(weighting='none', epsilon=3.61)
    assert_array_almost_equal(ngraph(self.pts, symmetrize=False, **kw), exp)
    assert_array_almost_equal(ngraph(self.pts, symmetrize=True, **kw),
                              (exp+exp.T)/2)

  def test_precomputed(self):
    D = pairwise_distances(self.pts, metric='l2')
    expected = np.array([[0,1,1,1],[1,0,1,1],[1,1,0,0],[1,1,0,0]])
    actual = ngraph(D, precomputed=True, k=2)
    assert_array_almost_equal(actual, expected, decimal=4)


if __name__ == '__main__':
  unittest.main()
