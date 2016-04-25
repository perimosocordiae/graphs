import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import locally_linear_embedding
from graphs import Graph
from graphs.construction import neighbor_graph


def assert_signless_array_almost_equal(a, b, **kwargs):
  a = np.asarray(a)
  b = np.asarray(b)
  if (a.flat[0] < 0 and b.flat[0] > 0) or (a.flat[0] > 0 and b.flat[0] < 0):
    assert_array_almost_equal(a, -b, **kwargs)
  else:
    assert_array_almost_equal(a, b, **kwargs)


class TestEmbeddings(unittest.TestCase):
  def test_isomap(self):
    expected = [-np.sqrt(8), -np.sqrt(2), 0, np.sqrt(2), np.sqrt(8)]
    G = Graph.from_adj_matrix([[0, np.sqrt(2), 2.82842712, 0, 0],
                               [np.sqrt(2), 0, np.sqrt(2), 0, 0],
                               [0, np.sqrt(2), 0, np.sqrt(2), 0],
                               [0, 0, np.sqrt(2), 0, np.sqrt(2)],
                               [0, 0, 2.82842712, np.sqrt(2), 0]])
    Y = G.isomap(num_vecs=1)
    self.assertEqual(Y.shape, (5, 1))
    assert_array_almost_equal(Y[:,0], expected)

  def test_laplacian_eigenmaps(self):
    # Test a simple chain graph
    expected = np.array([0.5, 0.5, 0., -0.5, -0.5])
    W = np.zeros((5,5)) + np.diag(np.ones(4), k=1) + np.diag(np.ones(4), k=-1)
    G = Graph.from_adj_matrix(W)
    Y = G.laplacian_eigenmaps(num_vecs=1)
    self.assertEqual(Y.shape, (5, 1))
    assert_signless_array_almost_equal(Y[:,0], expected)
    # Test num_vecs=None case
    Y = G.laplacian_eigenmaps()
    self.assertEqual(Y.shape, (5, 4))
    assert_signless_array_almost_equal(Y[:,0], expected)
    # Test sparse case
    G = Graph.from_adj_matrix(csr_matrix(W))
    Y = G.laplacian_eigenmaps(num_vecs=1)
    self.assertEqual(Y.shape, (5, 1))
    assert_signless_array_almost_equal(Y[:,0], expected)

  def test_locality_preserving_projections(self):
    X = np.array([[1,2],[2,1],[3,1.5],[4,0.5],[5,1]])
    G = Graph.from_adj_matrix([[0, 1, 1, 0, 0],
                               [1, 0, 1, 0, 0],
                               [1, 1, 0, 1, 1],
                               [0, 0, 1, 0, 1],
                               [0, 0, 1, 1, 0]])
    proj = G.locality_preserving_projections(X, num_vecs=1)
    assert_array_almost_equal(proj, np.array([[-0.95479113],[0.29727749]]))
    # test case with bigger d than n
    X = np.hstack((X, X))[:3]
    G = Graph.from_adj_matrix(G.matrix()[:3,:3])
    proj = G.locality_preserving_projections(X, num_vecs=1)
    assert_array_almost_equal(proj, np.array([[0.9854859,0.1697574,0,0]]).T)

  def test_locally_linear_embedding(self):
    np.random.seed(1234)
    pts = np.random.random((5, 3))
    expected = locally_linear_embedding(pts, 3, 1)[0]
    G = neighbor_graph(pts, k=3).barycenter_edge_weights(pts, copy=False)
    actual = G.locally_linear_embedding(num_vecs=1)
    assert_signless_array_almost_equal(expected, actual)

  def test_neighborhood_preserving_embedding(self):
    X = np.array([[1,2],[2,1],[3,1.5],[4,0.5],[5,1]])
    G = Graph.from_adj_matrix([[0, 1, 1, 0, 0],
                               [1, 0, 1, 0, 0],
                               [1, 1, 0, 1, 1],
                               [0, 0, 1, 0, 1],
                               [0, 0, 1, 1, 0]])
    proj = G.neighborhood_preserving_embedding(X, num_vecs=1)
    assert_signless_array_almost_equal(proj, [[0.99763], [0.068804]])

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
    actual = G.laplacian_pca(mX, num_vecs=1, beta=0)[:,:1]
    self.assertEqual(expected.shape, actual.shape)
    assert_signless_array_almost_equal(expected[:,0], actual[:,0], decimal=1)

  def test_circular_layout(self):
    G = Graph.from_edge_pairs([], num_vertices=4)
    expected = np.array([[1,0],[0,1],[-1,0],[0,-1]])
    assert_array_almost_equal(G.layout_circle(), expected)
    # edge cases
    for nv in (0, 1):
      G = Graph.from_edge_pairs([], num_vertices=nv)
      X = G.layout_circle()
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
    assert_array_almost_equal(G.layout_spring(), expected)
    # Test initial_layout kwarg
    X = np.arange(10).reshape((5,2))
    expected = np.array([
        [1.837091, 2.837091],
        [2.691855, 3.691855],
        [3.396880, 4.396880],
        [5.307083, 6.307083],
        [6.162909, 7.162909]])
    assert_array_almost_equal(G.layout_spring(initial_layout=X), expected)

if __name__ == '__main__':
  unittest.main()
