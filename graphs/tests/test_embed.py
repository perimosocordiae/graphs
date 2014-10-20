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
    Y = embed.laplacian_eigenmaps(Graph.from_adj_matrix(W), num_vecs=1)
    self.assertEqual(Y.shape, (5, 1))
    assert_array_almost_equal(Y[:,0], expected)
    # Test sparse case + return_vals
    S = csr_matrix(W)
    Y, vals = embed.laplacian_eigenmaps(Graph.from_adj_matrix(S), num_vecs=1,
                                        return_vals=True)
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


if __name__ == '__main__':
  unittest.main()
