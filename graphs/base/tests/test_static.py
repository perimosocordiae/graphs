import unittest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from scipy.sparse import csr_matrix

from graphs.base import Graph

PAIRS = np.array([[0,1],[0,2],[1,1],[2,1],[3,3]])
ADJ = [[0,1,1,0],
       [0,1,0,0],
       [0,1,0,0],
       [0,0,0,1]]


class TestStaticConstructors(unittest.TestCase):
  def test_from_pairs(self):
    g = Graph.from_edge_pairs(PAIRS)
    self.assertEqual(g.num_edges(), 5)
    self.assertEqual(g.num_vertices(), 4)
    g = Graph.from_edge_pairs(PAIRS, num_vertices=10)
    self.assertEqual(g.num_edges(), 5)
    self.assertEqual(g.num_vertices(), 10)
    g = Graph.from_edge_pairs(PAIRS, symmetric=True)
    self.assertEqual(g.num_edges(), 8)
    self.assertEqual(g.num_vertices(), 4)

  def test_from_pairs_empty(self):
    g = Graph.from_edge_pairs([])
    self.assertEqual(g.num_edges(), 0)
    self.assertEqual(g.num_vertices(), 0)
    ii, jj = g.pairs().T
    assert_array_equal(ii, [])
    assert_array_equal(jj, [])
    # Make sure ii and jj have indexable dtypes
    PAIRS[ii,jj]
    # Make sure num_vertices is set correctly
    g = Graph.from_edge_pairs([], num_vertices=5)
    self.assertEqual(g.num_edges(), 0)
    self.assertEqual(g.num_vertices(), 5)

  def test_from_pairs_floating(self):
    g = Graph.from_edge_pairs(PAIRS.astype(float))
    p = g.pairs()
    self.assertTrue(np.can_cast(p, PAIRS.dtype, casting='same_kind'),
                    "Expected integral dtype, got %s" % p.dtype)
    assert_array_equal(p, PAIRS)

  def test_from_pairs_weighted(self):
    w = np.array([1,1,0.1,2,1,2,3.1,4])
    p = [[0,1],[1,2],[2,3],[3,4],[1,0],[2,1],[3,2],[4,3]]
    expected = [[0,1,0,0,0],[1,0,1,0,0],[0,2,0,0.1,0],[0,0,3.1,0,2],[0,0,0,4,0]]
    G = Graph.from_edge_pairs(p, weights=w, num_vertices=5)
    assert_array_almost_equal(G.matrix(dense=True), expected)

    # weighted + symmetric
    expected = [[0,1,2,0],[1,3,4,0],[2,4,0,0],[0,0,0,5]]
    G = Graph.from_edge_pairs(PAIRS, symmetric=True, weights=np.arange(1,6))
    assert_array_equal(G.matrix(dense=True), expected)

  def test_from_adj(self):
    m = Graph.from_adj_matrix(ADJ)
    self.assertEqual(m.num_edges(), 5)
    self.assertEqual(m.num_vertices(), 4)
    m = Graph.from_adj_matrix(csr_matrix(ADJ))
    self.assertEqual(m.num_edges(), 5)
    self.assertEqual(m.num_vertices(), 4)

if __name__ == '__main__':
  unittest.main()
