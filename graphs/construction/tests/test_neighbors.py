import unittest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from sklearn.metrics.pairwise import pairwise_distances

from graphs.construction import neighbors


def ngraph(*a, **k):
    return neighbors.neighbor_graph(*a,**k).matrix(dense=True)


class TestNeighbors(unittest.TestCase):
  def setUp(self):
    self.pts = np.array([[0,0],[1,2],[3,2],[-1,0]])
    self.bin_adj = np.array([[0,1,0,1],[1,0,1,0],[1,1,0,0],[1,1,0,0]])
    self.l2_adj = np.sqrt([[0,5,0,1],[5,0,4,0],[13,4,0,0],[1,8,0,0]])

  def test_neighbor_graph(self):
    self.assertRaises(AssertionError, ngraph, self.pts)

  def test_binary_weighting(self):
    assert_array_equal(ngraph(self.pts, weighting='binary', k=2), self.bin_adj)
    assert_array_equal(ngraph(self.pts, weighting='binary', k=2, epsilon=100),
                       self.bin_adj)
    # Add extra values for e-ball
    self.bin_adj[0,2] = 1
    self.bin_adj[1,3] = 1
    assert_array_equal(ngraph(self.pts, weighting='binary', epsilon=3.61),
                       self.bin_adj)

  def test_no_weighting(self):
    assert_array_almost_equal(ngraph(self.pts, k=2), self.l2_adj)
    # Add extra values for e-ball
    self.l2_adj[0,2] = np.sqrt(13)
    self.l2_adj[1,3] = np.sqrt(8)
    assert_array_almost_equal(ngraph(self.pts, epsilon=3.61), self.l2_adj)

  def test_precomputed(self):
    D = pairwise_distances(self.pts, metric='l2')
    actual = ngraph(D, precomputed=True, k=2)
    assert_array_almost_equal(actual, self.l2_adj, decimal=4)
    actual = ngraph(D, precomputed=True, k=2, weighting='binary')
    assert_array_almost_equal(actual, self.bin_adj, decimal=4)

  def test_nearest_neighbors(self):
    nns = neighbors.nearest_neighbors
    pt = np.zeros(2)
    self.assertRaises(AssertionError, nns, pt, self.pts)
    assert_array_equal(nns(pt, self.pts, k=2), [[0,3]])
    assert_array_equal(nns(pt, self.pts, epsilon=2), [[0,3]])
    assert_array_equal(nns(pt, self.pts, k=2, epsilon=10), [[0,3]])
    # Check return_dists
    dists, inds = nns(pt, self.pts, k=2, return_dists=True)
    assert_array_equal(inds, [[0,3]])
    assert_array_almost_equal(dists, [[0, 1]])
    dists, inds = nns(pt, self.pts, epsilon=2, return_dists=True)
    assert_array_equal(inds, [[0,3]])
    assert_array_almost_equal(dists, [[0, 1]])
    # Check precomputed
    D = pairwise_distances(pt[None], self.pts, metric='l1')
    self.assertRaises(AssertionError, nns, pt, self.pts, precomputed=True, k=2)
    assert_array_equal(nns(D, precomputed=True, k=2), [[0,3]])
    # Check 2d query shape
    pt = [[0,0]]
    assert_array_equal(nns(pt, self.pts, k=2), [[0,3]])
    # Check all-pairs mode
    assert_array_equal(nns(self.pts, k=2), [[0,3],[1,2],[2,1],[3,0]])


if __name__ == '__main__':
  unittest.main()
