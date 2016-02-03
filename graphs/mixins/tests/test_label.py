import unittest
import numpy as np
from numpy.testing import assert_array_equal
from scipy.sparse import coo_matrix, issparse
from sklearn.metrics.cluster import adjusted_rand_score
from graphs import Graph
from graphs.construction import neighbor_graph

PAIRS = np.array([[0,1],[0,2],[1,0],[1,2],[2,0],[2,1],[3,4],[4,3]])
ADJ = [[0,1,1,0,0],
       [1,0,1,0,0],
       [1,1,0,0,0],
       [0,0,0,0,1],
       [0,0,0,1,0]]


class TestLabel(unittest.TestCase):
  def setUp(self):
    self.graphs = [
        Graph.from_edge_pairs(PAIRS),
        Graph.from_adj_matrix(ADJ),
        Graph.from_adj_matrix(coo_matrix(ADJ)),
    ]

  def test_greedy_coloring(self):
    for G in self.graphs:
      assert_array_equal([1,2,3,1,2], G.greedy_coloring())

  def test_kernel_matrix(self):
    for G in self.graphs:
      for kernel in ('none', 'binary'):
        K = G._kernel_matrix(kernel)
        if issparse(K):
          K = K.toarray()
        assert_array_equal(K, ADJ)
      self.assertRaises(ValueError, G._kernel_matrix, 'foobar')

  def test_spectral_clustering(self):
    pts = np.random.random(size=(20, 2))
    pts[10:] += 2
    expected = np.zeros(20)
    expected[10:] = 1
    G = neighbor_graph(pts, k=11).symmetrize()

    labels = G.spectral_clustering(2, kernel='rbf')
    self.assertGreater(adjusted_rand_score(expected, labels), 0.95)

  def test_spread_labels(self):
    pts = np.random.random(size=(20, 2))
    pts[10:] += 2
    expected = np.zeros(20)
    expected[10:] = 1
    partial = expected.copy()
    partial[1:-1] = -1
    G = neighbor_graph(pts, k=11).symmetrize()

    labels = G.spread_labels(partial, kernel='rbf', alpha=0.2, tol=1e-3,
                             max_iter=30)
    self.assertGreater(adjusted_rand_score(expected, labels), 0.95)

  def test_regression(self):
    t = np.linspace(0, 1, 31)
    pts = np.column_stack((np.sin(t), np.cos(t)))
    G = neighbor_graph(pts, k=3).symmetrize()
    y_mask = slice(None, None, 2)

    # test the interpolated case
    x = G.regression(t[y_mask], y_mask)
    assert_array_equal(t, np.linspace(0, 1, 31))  # ensure t hasn't changed
    self.assertLess(np.linalg.norm(t - x), 0.15)

    # test the penalized case
    x = G.regression(t[y_mask], y_mask, smoothness_penalty=1e-4)
    self.assertLess(np.linalg.norm(t - x), 0.15)

if __name__ == '__main__':
  unittest.main()
