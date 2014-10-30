import matplotlib
matplotlib.use('template')

import unittest
import numpy as np
from numpy.testing import assert_array_equal
from scipy.sparse import csr_matrix

from graphs.base import (
    Graph, EdgePairGraph, DenseAdjacencyMatrixGraph, SparseAdjacencyMatrixGraph)

PAIRS = np.array([[0,1],[0,2],[1,1],[2,1],[3,3]])
ADJ = [[0,1,1,0],
       [0,1,0,0],
       [0,1,0,0],
       [0,0,0,1]]


class TestStaticConstructors(unittest.TestCase):
  def test_from_pairs(self):
    epg = Graph.from_edge_pairs(PAIRS)
    self.assertEqual(epg.num_edges(), 5)
    self.assertEqual(epg.num_vertices(), 4)
    epg = Graph.from_edge_pairs(PAIRS, num_vertices=10)
    self.assertEqual(epg.num_edges(), 5)
    self.assertEqual(epg.num_vertices(), 10)

  def test_from_adj(self):
    m = Graph.from_adj_matrix(ADJ)
    self.assertEqual(m.num_edges(), 5)
    self.assertEqual(m.num_vertices(), 4)
    m = Graph.from_adj_matrix(csr_matrix(ADJ))
    self.assertEqual(m.num_edges(), 5)
    self.assertEqual(m.num_vertices(), 4)


class TestEdgePairGraph(unittest.TestCase):
  def setUp(self):
    self.epg = EdgePairGraph(PAIRS)

  def test_epg_pairs(self):
    self.assert_(self.epg.pairs(copy=False) is PAIRS)
    P = self.epg.pairs(copy=True)
    self.assert_(P is not PAIRS)
    assert_array_equal(P, PAIRS)

  def test_epg_matrix(self):
    M = self.epg.matrix()
    assert_array_equal(M.toarray(), ADJ)
    M = self.epg.matrix(dense=True)
    assert_array_equal(M, ADJ)
    M = self.epg.matrix(csr=True)
    self.assertEqual(M.format, 'csr')
    assert_array_equal(M.toarray(), ADJ)


class TestAdjacencyMatrixGraphs(unittest.TestCase):
  def setUp(self):
    self.G = DenseAdjacencyMatrixGraph(ADJ)
    self.S = SparseAdjacencyMatrixGraph(csr_matrix(ADJ))

  def test_pairs(self):
    assert_array_equal(self.G.pairs(), PAIRS)
    assert_array_equal(self.S.pairs(), PAIRS)

  def test_matrix(self):
    M = self.G.matrix()
    assert_array_equal(M, ADJ)
    M = self.G.matrix(csr=True)
    self.assertEqual(M.format, 'csr')
    assert_array_equal(M.toarray(), ADJ)
    M = self.S.matrix()
    self.assertEqual(M.format, 'csr')
    assert_array_equal(M.toarray(), ADJ)
    M = self.G.matrix(dense=True)
    assert_array_equal(M, ADJ)


class TestGenericMembers(unittest.TestCase):
  def setUp(self):
    spadj = csr_matrix(ADJ)
    spadj[0,0] = 0  # Add an explicit zero
    self.graphs = [
        EdgePairGraph(PAIRS),
        DenseAdjacencyMatrixGraph(ADJ),
        SparseAdjacencyMatrixGraph(spadj)
    ]
    self.weighted = DenseAdjacencyMatrixGraph(np.array(ADJ)*np.arange(4)[None])

  def test_properties(self):
    for G in self.graphs:
      self.assertEqual(G.num_edges(), 5, 'num_edges (%s)' % type(G))
      self.assertEqual(G.num_vertices(), 4, 'num_vertices (%s)' % type(G))

  def test_degree(self):
    for G in self.graphs:
      in_degree = G.degree('in', unweighted=True)
      out_degree = G.degree('out', unweighted=True)
      assert_array_equal(in_degree, [0, 3, 1, 1])
      assert_array_equal(out_degree, [2, 1, 1, 1])

  def test_degree_weighted(self):
    in_degree = self.weighted.degree(kind='in', unweighted=False)
    out_degree = self.weighted.degree(kind='out', unweighted=False)
    assert_array_equal(in_degree, [0, 3, 2, 3])
    assert_array_equal(out_degree, [3, 1, 1, 3])

  def test_adj_list(self):
    expected = [[1,2],[1],[1],[3]]
    for G in self.graphs:
      adj_list = G.adj_list()
      for a,e in zip(adj_list, expected):
        assert_array_equal(a, e)

  def test_add_self_edges(self):
    expected = (np.array(ADJ) + np.eye(len(ADJ))).astype(bool).astype(int)
    for G in self.graphs:
      gg = G.add_self_edges()
      assert_array_equal(gg.matrix(dense=True), expected,
                         'unweighted (%s)' % type(G))
    expected = np.array(ADJ, dtype=float)
    np.fill_diagonal(expected, 0.5)
    for G in self.graphs:
      if G.is_weighted():
        gg = G.add_self_edges(weight=0.5)
        assert_array_equal(gg.matrix(dense=True), expected,
                           'weighted (%s)' % type(G))

  def test_symmetrize(self):
    adj = np.array(ADJ)
    bool_expected = np.logical_or(adj, adj.T)
    # max
    expected = np.maximum(adj, adj.T)
    self._help_test_symmetrize(expected, bool_expected, 'max')
    # sum
    expected = adj + adj.T
    self._help_test_symmetrize(expected, bool_expected, 'sum')
    # avg
    expected = expected.astype(float) / 2
    self._help_test_symmetrize(expected, bool_expected, 'avg')

  def _help_test_symmetrize(self, expected, bool_expected, method):
    for G in self.graphs:
      sym = G.symmetrize(overwrite=False, method=method).matrix(dense=True)
      msg = '%s symmetrize (%s)' % (method, type(G))
      if G.is_weighted():
        assert_array_equal(sym, expected, msg)
      else:
        assert_array_equal(sym, bool_expected, msg)

  def test_edge_weights(self):
    expected = np.ones(5)
    for G in self.graphs:
      if G.is_weighted():
        ew = G.edge_weights()
        assert_array_equal(ew, expected, 'edge weights (%s)' % type(G))
    expected = [1,2,1,1,3]
    assert_array_equal(self.weighted.edge_weights(), expected)

if __name__ == '__main__':
  unittest.main()
