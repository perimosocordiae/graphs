import matplotlib
matplotlib.use('template')

import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from graphs import embed, Graph


class TestEmbeddings(unittest.TestCase):
  def test_isomap(self):
    expected = [-np.sqrt(8), -np.sqrt(2), 0, np.sqrt(2), np.sqrt(8)]
    G = Graph.from_adj_matrix([[0, np.sqrt(2), 2.82842712, 0, 0],
                               [np.sqrt(2), 0, np.sqrt(2), 0, 0],
                               [0, np.sqrt(2), 0, np.sqrt(2), 0],
                               [0, 0, np.sqrt(2), 0, np.sqrt(2)],
                               [0, 0, 2.82842712, np.sqrt(2), 0]])
    Y = embed.isomap(G, num_vecs=1)
    self.assertEqual(Y.shape, (5, 1))
    assert_array_almost_equal(Y[:,0], expected)

  def test_laplacian_eigenmaps(self):
    # Test a simple chain graph
    expected = np.array([0.5, 0.5, 0., -0.5, -0.5])
    W = np.zeros((5,5)) + np.diag(np.ones(4), k=1) + np.diag(np.ones(4), k=-1)
    G = Graph.from_adj_matrix(W)
    Y = embed.laplacian_eigenmaps(G, num_vecs=1)
    self.assertEqual(Y.shape, (5, 1))
    assert_array_almost_equal(Y[:,0], expected)
    # Test num_vecs=None case
    Y = embed.laplacian_eigenmaps(G)
    self.assertEqual(Y.shape, (5, 4))
    assert_array_almost_equal(Y[:,0], expected)
    # Test sparse case + return_vals
    G = Graph.from_adj_matrix(csr_matrix(W))
    Y, vals = embed.laplacian_eigenmaps(G, num_vecs=1, return_vals=True)
    assert_array_almost_equal(vals, [0.292893])
    self.assertEqual(Y.shape, (5, 1))
    assert_array_almost_equal(Y[:,0], expected)

  def test_locality_preserving_projections(self):
    X = np.array([[1,2],[2,1],[3,1.5],[4,0.5],[5,1]])
    G = Graph.from_adj_matrix([[0, 1, 1, 0, 0],
                               [1, 0, 1, 0, 0],
                               [1, 1, 0, 1, 1],
                               [0, 0, 1, 0, 1],
                               [0, 0, 1, 1, 0]])
    proj = embed.locality_preserving_projections(G, X, num_vecs=1)
    assert_array_almost_equal(proj, np.array([[-0.95479113],[0.29727749]]))
    # test case with bigger d than n
    X = np.hstack((X, X))[:3]
    G = Graph.from_adj_matrix(G.matrix()[:3,:3])
    proj = embed.locality_preserving_projections(G, X, num_vecs=1)
    assert_array_almost_equal(proj, np.array([[0.9854859,0.1697574,0,0]]).T)

  def test_laplacian_pca(self):
    X = np.array([[1,2],[2,1],[3,1.5],[4,0.5],[5,1]])
    G = Graph.from_adj_matrix([[0, 1, 1, 0, 0],
                               [1, 0, 1, 0, 0],
                               [1, 1, 0, 1, 1],
                               [0, 0, 1, 0, 1],
                               [0, 0, 1, 1, 0]])
    # check that beta=0 gets the (roughly) the same answer as PCA
    mX = X - X.mean(axis=0)
    expected = PCA(n_components=1).fit_transform(mX)
    actual = embed.laplacian_pca(G, mX, num_vecs=1, beta=0)[:,:1]
    self.assertTrue(np.abs(expected - actual).sum() < 0.5)

  def test_circular_layout(self):
    G = Graph.from_edge_pairs([], num_vertices=4)
    expected = np.array([[1,0],[0,1],[-1,0],[0,-1]])
    assert_array_almost_equal(embed.circular_layout(G), expected)
    # edge cases
    for nv in (0, 1):
      G = Graph.from_edge_pairs([], num_vertices=nv)
      X = embed.circular_layout(G)
      self.assertEqual(X.shape, (nv, 2))

  def test_spring_layout(self):
    np.random.seed(1234)
    w = np.array([1,2,0.1,1,1,2,0.1,1])
    p = [[0,1],[1,2],[2,3],[3,4],[1,0],[2,1],[3,2],[4,3]]
    G = Graph.from_edge_pairs(p, weights=w, num_vertices=5)
    expected = np.array([
        [-1.12951518, 0.44975598],
        [-0.42574481, 0.51702804],
        [0.58946761,  0.61403187],
        [0.96513010,  0.64989485],
        [1.67011322,  0.71714073]])
    assert_array_almost_equal(embed.spring_layout(G), expected)

if __name__ == '__main__':
  unittest.main()
