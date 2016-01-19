import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal
from sklearn.metrics import pairwise_distances

from graphs.construction import neighbor_graph
from graphs.construction.incremental import incremental_neighbor_graph
from graphs.mini_six import zip_longest, range

np.set_printoptions(precision=3, suppress=True)


def ngraph(*a, **k):
    return neighbor_graph(*a,**k).matrix(dense=True)


class TestNeighbors(unittest.TestCase):
  def setUp(self):
    self.pts = np.array([[0,0],[1,2],[3,2.5],[-1,0],[.5,.2],[3,.6],[-2,-0.5]])

  def test_k_range(self):
    k_range = range(1, 5)
    incr_gen = incremental_neighbor_graph(self.pts, k=k_range)
    for k, G in zip_longest(k_range, incr_gen):
      expected = ngraph(self.pts, k=k)
      assert_array_almost_equal(G.matrix(dense=True), expected)

    # non-uniform steps
    k_range = [1, 3, 6]
    incr_gen = incremental_neighbor_graph(self.pts, k=k_range)
    for k, G in zip_longest(k_range, incr_gen):
      expected = ngraph(self.pts, k=k)
      assert_array_almost_equal(G.matrix(dense=True), expected)

  def test_eps_range(self):
    eps_range = np.linspace(0.1, 5.5, 5)
    incr_gen = incremental_neighbor_graph(self.pts, epsilon=eps_range)
    for eps, G in zip_longest(eps_range, incr_gen):
      expected = ngraph(self.pts, epsilon=eps)
      assert_array_almost_equal(G.matrix(dense=True), expected)

  def test_k_eps_range(self):
    # varied k with fixed epsilon
    k_range = range(1, 5)
    incr_gen = incremental_neighbor_graph(self.pts, k=k_range, epsilon=3.)
    for k, G in zip_longest(k_range, incr_gen):
      expected = ngraph(self.pts, k=k, epsilon=3.)
      assert_array_almost_equal(G.matrix(dense=True), expected)

    # varied eps with fixed k
    eps_range = np.linspace(0.1, 5.5, 5)
    incr_gen = incremental_neighbor_graph(self.pts, k=3, epsilon=eps_range)
    for eps, G in zip_longest(eps_range, incr_gen):
      expected = ngraph(self.pts, k=3, epsilon=eps)
      assert_array_almost_equal(G.matrix(dense=True), expected)

  def test_l1_precomputed(self):
    dist = pairwise_distances(self.pts, metric='l1')
    k_range = range(1, 5)
    incr_gen = incremental_neighbor_graph(dist, precomputed=True, k=k_range)
    for k, G in zip_longest(k_range, incr_gen):
      expected = ngraph(dist, precomputed=True, k=k)
      assert_array_almost_equal(G.matrix(dense=True), expected)


if __name__ == '__main__':
  unittest.main()
