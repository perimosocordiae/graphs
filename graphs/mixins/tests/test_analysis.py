import unittest
import numpy as np
import warnings
from numpy.testing import assert_array_equal, assert_array_almost_equal
from scipy.sparse import coo_matrix
from graphs import Graph

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
      self.assertEqual(str(w[0].message),
                       'phi failed to converge after 2 iterations')

  def test_bandwidth(self):
    for G in self.graphs:
      self.assertEqual(G.bandwidth(), 2)

  def test_profile(self):
    for G in self.graphs:
      self.assertEqual(G.profile(), 1)

  def test_betweenness(self):
    for G in self.graphs:
      G.symmetrize(copy=False)
      _test_btw(G, 'vertex', False, False, np.zeros(5))
      _test_btw(G, 'vertex', False, True, np.zeros(5))
      _test_btw(G, 'edge', False, False, np.ones(8)/2.)
      _test_btw(G, 'edge', False, True, np.ones(8))
      if G.is_weighted():
        _test_btw(G, 'vertex', True, False, [0,0.5,0,0,0])
        _test_btw(G, 'vertex', True, True, [0,1,0,0,0])
        _test_btw(G, 'edge', True, False, np.array([3,1,3,3,1,3,2,2])/4.)
        _test_btw(G, 'edge', True, True, np.array([3,1,3,3,1,3,2,2])/2.)

  def test_betweenness_weighted(self):
    # test a weighted graph with different kinds of weights
    G = Graph.from_adj_matrix([[0,1,2,0],[1,0,0,3],[2,0,0,1],[0,3,1,0]])
    _test_btw(G, 'vertex', False, False, [0.5]*4)
    _test_btw(G, 'vertex', False, True, [1]*4)
    _test_btw(G, 'vertex', True, False, [1,0,1,0])
    _test_btw(G, 'vertex', True, True, [2,0,2,0])
    _test_btw(G, 'edge', False, False, [1,1,1,1,1,1,1,1])
    _test_btw(G, 'edge', False, True, [2,2,2,2,2,2,2,2])
    _test_btw(G, 'edge', True, False, np.array([2,3,2,1,3,2,1,2])/2.)
    _test_btw(G, 'edge', True, True, [2,3,2,1,3,2,1,2])

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


def _test_btw(G, k, w, d, exp):
  assert_array_equal(G.betweenness(kind=k, weighted=w, directed=d), exp)

if __name__ == '__main__':
  unittest.main()
