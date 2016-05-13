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

  def _make_blobs_graph(self, k=11):
    pts = np.random.random(size=(20, 2))
    pts[10:] += 2
    labels = np.zeros(20)
    labels[10:] = 1
    G = neighbor_graph(pts, k=k).symmetrize()
    return G, labels

  def test_greedy_coloring(self):
    for G in self.graphs:
      assert_array_equal([1,2,3,1,2], G.color_greedy())

  def test_kernel_matrix(self):
    for G in self.graphs:
      for kernel in ('none', 'binary'):
        K = G._kernel_matrix(kernel)
        if issparse(K):
          K = K.toarray()
        assert_array_equal(K, ADJ)
      self.assertRaises(ValueError, G._kernel_matrix, 'foobar')

  def test_spectral_clustering(self):
    G, expected = self._make_blobs_graph(k=11)
    labels = G.cluster_spectral(2, kernel='rbf')
    self.assertGreater(adjusted_rand_score(expected, labels), 0.95)

  def test_nn_classifier(self):
    G, expected = self._make_blobs_graph(k=4)
    partial = expected.copy()
    partial[1:-1] = -1

    labels = G.classify_nearest(partial)
    self.assertGreater(adjusted_rand_score(expected, labels), 0.95)

  def test_lgc_classifier(self):
    G, expected = self._make_blobs_graph(k=11)
    partial = expected.copy()
    partial[1:-1] = -1

    labels = G.classify_lgc(partial, kernel='rbf', alpha=0.2, tol=1e-3,
                            max_iter=30)
    self.assertGreater(adjusted_rand_score(expected, labels), 0.95)

  def test_harmonic_classifier(self):
    G, expected = self._make_blobs_graph(k=4)
    partial = expected.copy()
    partial[1:-1] = -1

    labels = G.classify_harmonic(partial, use_CMN=True)
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

    # test the boolean mask case
    y_mask = np.zeros_like(t, dtype=bool)
    y_mask[::2] = True
    x = G.regression(t[y_mask], y_mask)
    self.assertLess(np.linalg.norm(t - x), 0.15)

    # test the penalized case
    x = G.regression(t[y_mask], y_mask, smoothness_penalty=1e-4)
    self.assertLess(np.linalg.norm(t - x), 0.15)

    # test no kernel + dense laplacian case
    dG = Graph.from_adj_matrix(G.matrix(dense=True))
    x = dG.regression(t[y_mask], y_mask, kernel='none')
    self.assertLess(np.linalg.norm(t - x), 0.25)
    x = dG.regression(t[y_mask], y_mask, smoothness_penalty=1e-4, kernel='none')
    self.assertLess(np.linalg.norm(t - x), 0.25)

    # test the multidimensional regression case
    tt = np.column_stack((t, t[::-1]))
    x = G.regression(tt[y_mask], y_mask)
    self.assertLess(np.linalg.norm(tt - x), 0.2)

    # check for bad inputs
    with self.assertRaisesRegexp(ValueError, r'^Invalid shape of y array'):
      G.regression([], y_mask)

if __name__ == '__main__':
  unittest.main()
