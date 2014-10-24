import matplotlib
matplotlib.use('template')

import unittest
import numpy as np
from numpy.testing import assert_array_equal
from scipy.sparse import csr_matrix

from graphs.base import (
    Graph, EdgePairGraph, DenseAdjacencyMatrixGraph, SparseAdjacencyMatrixGraph)

PAIRS = np.array([[0,1],[0,2],[1,2],[3,4]])
ADJ = [[0,1,1,0,0],
       [0,0,1,0,0],
       [0,0,0,0,0],
       [0,0,0,0,1],
       [0,0,0,0,0]]


class TestStaticConstructors(unittest.TestCase):
  def test_from_pairs(self):
    epg = Graph.from_edge_pairs(PAIRS)
    self.assertEqual(epg.num_edges(), 4)
    self.assertEqual(epg.num_vertices(), 5)
    epg = Graph.from_edge_pairs(PAIRS, num_vertices=10)
    self.assertEqual(epg.num_edges(), 4)
    self.assertEqual(epg.num_vertices(), 10)

  def test_from_adj(self):
    m = Graph.from_adj_matrix(ADJ)
    self.assertEqual(m.num_edges(), 4)
    self.assertEqual(m.num_vertices(), 5)
    m = Graph.from_adj_matrix(csr_matrix(ADJ))
    self.assertEqual(m.num_edges(), 4)
    self.assertEqual(m.num_vertices(), 5)


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
    self.weighted = DenseAdjacencyMatrixGraph(np.array(ADJ)*np.arange(5)[None])

  def test_properties(self):
    for G in self.graphs:
      self.assertEqual(G.num_edges(), 4, 'num_edges (%s)' % type(G))
      self.assertEqual(G.num_vertices(), 5, 'num_vertices (%s)' % type(G))

  def test_degree(self):
    for G in self.graphs:
      in_degree = G.degree('in', unweighted=True)
      out_degree = G.degree('out', unweighted=True)
      assert_array_equal(in_degree, [0, 1, 2, 0, 1])
      assert_array_equal(out_degree, [2, 1, 0, 1, 0])

  def test_degree_weighted(self):
    in_degree = self.weighted.degree(kind='in', unweighted=False)
    out_degree = self.weighted.degree(kind='out', unweighted=False)
    assert_array_equal(in_degree, [0, 1, 4, 0, 4])
    assert_array_equal(out_degree, [3, 2, 0, 4, 0])

  def test_adj_list(self):
    expected = [[1,2],[2],[],[4],[]]
    for G in self.graphs:
      adj_list = G.adj_list()
      for a,e in zip(adj_list, expected):
        assert_array_equal(a, e)

  def test_add_self_edges(self):
    expected = np.array(ADJ) + np.eye(len(ADJ))
    for G in self.graphs:
      gg = G.add_self_edges()
      assert_array_equal(gg.matrix(dense=True), expected,
                         'unweighted (%s)' % type(G))
    expected = np.array(ADJ) + 0.5 * np.eye(len(ADJ))
    for G in self.graphs:
      if G.is_weighted():
        gg = G.add_self_edges(weight=0.5)
        assert_array_equal(gg.matrix(dense=True), expected,
                           'weighted (%s)' % type(G))

  def test_symmetrize(self):
    expected = np.array(ADJ)
    # sum
    expected += expected.T
    for G in self.graphs:
      sym = G.symmetrize(overwrite=False, method='sum')
      assert_array_equal(sym.matrix(dense=True), expected,
                         'sum symmetrize (%s)' % type(G))
    # avg
    expected = expected.astype(float) / 2
    for G in self.graphs:
      sym = G.symmetrize(overwrite=False, method='avg')
      assert_array_equal(sym.matrix(dense=True), expected,
                         'avg symmetrize (%s)' % type(G))
    # max
    expected = np.maximum(np.array(ADJ), np.array(ADJ).T)
    for G in self.graphs:
      sym = G.symmetrize(overwrite=False, method='max')
      assert_array_equal(sym.matrix(dense=True), expected,
                         'max symmetrize (%s)' % type(G))

  def test_edge_weights(self):
    expected = np.ones(4)
    for G in self.graphs:
      if G.is_weighted():
        ew = G.edge_weights()
        assert_array_equal(ew, expected, 'edge weights (%s)' % type(G))
    expected = [1,2,2,4]
    assert_array_equal(self.weighted.edge_weights(), expected)

if __name__ == '__main__':
  unittest.main()
