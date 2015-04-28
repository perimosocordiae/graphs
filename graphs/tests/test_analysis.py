import unittest
import numpy as np
import warnings
from numpy.testing import assert_array_equal, assert_array_almost_equal
from scipy.sparse import coo_matrix
from graphs import Graph

try:
  import igraph
  HAS_IGRAPH = True
except ImportError:
  HAS_IGRAPH = False

PAIRS = np.array([[0,1],[0,2],[1,2],[2,0],[3,4],[4,3]])
ADJ = [[0,1,1,0,0],
       [0,0,1,0,0],
       [1,0,0,0,0],
       [0,0,0,0,1],
       [0,0,0,1,0]]


class TestAnalysis(unittest.TestCase):
  def setUp(self):
    self.graphs = [
        Graph.from_edge_pairs(PAIRS),
        Graph.from_adj_matrix(ADJ),
        Graph.from_adj_matrix(coo_matrix(ADJ)),
    ]

  def test_connected_components(self):
    for G in self.graphs:
      n, labels = G.connected_components()
      self.assertEqual(2, n)
      assert_array_equal(labels, [0,0,0,1,1])

  def test_greedy_coloring(self):
    for G in self.graphs:
      assert_array_equal([1,2,3,1,2], G.symmetrize().greedy_coloring())

  def test_ave_laplacian(self):
    g = Graph.from_adj_matrix([[0,1,2],[1,0,0],[2,0,0]])
    expected = np.array([[1,-0.5,0],[-0.5,1,0],[0,0,1]])
    assert_array_almost_equal(g.ave_laplacian(), expected)

  def test_directed_laplacian(self):
    expected = np.array([
        [0.239519, -0.05988, -0.179839, 0,   0],
        [-0.05988,  0.120562,-0.060281, 0,   0],
        [-0.179839,-0.060281, 0.239919, 0,   0],
        [0,         0,        0,        0.2,-0.2],
        [0,         0,        0,       -0.2, 0.2]])
    for G in self.graphs:
      L = G.directed_laplacian()
      assert_array_almost_equal(L, expected)

    # test non-convergence case
    with warnings.catch_warnings(record=True) as w:
      self.graphs[0].directed_laplacian(max_iter=2)
      self.assertEqual(len(w), 1)
      self.assertEqual(w[0].message.message,
                       'phi failed to converge after 2 iterations')

  def test_bandwidth(self):
    for G in self.graphs:
      self.assertEqual(G.bandwidth(), 2)

  def test_profile(self):
    for G in self.graphs:
      self.assertEqual(G.profile(), 1)

  @unittest.skipIf(not HAS_IGRAPH, 'betweenness requires igraph dependency')
  def test_betweenness(self):
    for G in self.graphs:
      assert_array_equal(G.betweenness(kind='vertex'), [1,0,1,0,0])
      assert_array_equal(G.betweenness(kind='edge'), [2,1,2,3,1,1])

  def test_eccentricity(self):
    for G in self.graphs:
      # unconnected graphs have infinite eccentricity
      assert_array_equal(G.eccentricity(), np.inf+np.ones(5))
    g = Graph.from_adj_matrix([[0,1,2],[1,0,0],[2,0,0]])
    assert_array_equal(g.eccentricity(), [2,3,3])

  def test_diameter(self):
    for G in self.graphs:
      # unconnected graphs have infinite diameter
      self.assertEqual(G.diameter(), np.inf)
    g = Graph.from_adj_matrix([[0,1,2],[1,0,0],[2,0,0]])
    self.assertEqual(g.diameter(), 3)

  def test_radius(self):
    for G in self.graphs:
      # unconnected graphs have infinite radius
      self.assertEqual(G.radius(), np.inf)
    g = Graph.from_adj_matrix([[0,1,2],[1,0,0],[2,0,0]])
    self.assertEqual(g.radius(), 2)

if __name__ == '__main__':
  unittest.main()
