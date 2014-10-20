import unittest
import numpy as np
from numpy.testing import assert_array_equal
from scipy.sparse import csr_matrix, coo_matrix

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

  def test_epg_simple(self):
    self.assertEqual(self.epg.num_edges(), 4)
    self.assertEqual(self.epg.num_vertices(), 5)

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


class TestGenericMembers(unittest.TestCase):
  def setUp(self):
    self.graphs = [
        EdgePairGraph(PAIRS),
        DenseAdjacencyMatrixGraph(ADJ),
        SparseAdjacencyMatrixGraph(coo_matrix(ADJ)),
    ]

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

if __name__ == '__main__':
  unittest.main()




  # def test_pairs_to_matrix(self):
  #   self.assertIsNone(self.corr._matrix)
  #   m = self.corr.matrix()
  #   self.assertEqual(len(m.shape), 2)
  #   self.assertEqual(m.shape[0], m.shape[1])
  #   d = self.corr.matrix(dense=True)
  #   assert_array_equal(d, self.m)

  # def test_identity(self):
  #   pairs = self.corr.pairs()
  #   matrix = self.corr.matrix()
  #   assert_array_equal(pairs, graph.Graph(matrix=matrix).pairs())
  #   assert_array_equal(matrix.A, graph.Graph(pairs=pairs).matrix().A)

  # def test_num_connected_components(self):
  #   self.assertEqual(2, self.corr.num_connected_components())
  #   p = numpy.vstack((self.p, [2,3]))
  #   c2 = graph.Graph(pairs=p)
  #   self.assertEqual(1, c2.num_connected_components())
  #   p = numpy.array([[0,1],[2,3],[4,5]])
  #   c3 = graph.Graph(pairs=p)
  #   self.assertEqual(3, c3.num_connected_components())

  # def test_num_edges(self):
  #   self.assertEqual(4, self.corr.num_edges())
  #   p = numpy.vstack((self.p, [2,3]))
  #   c2 = graph.Graph(pairs=p)
  #   self.assertEqual(5, c2.num_edges())
  #   m = numpy.ones((3,3))
  #   c3 = graph.Graph(matrix=m)
  #   self.assertEqual(9, c3.num_edges())

  # def test_adj_list(self):
  #   expected = [[1,2],[2],[],[4],[]]
  #   assert_array_equal(map(list,self.corr.adj_list()), expected)

  # def test_symmetrize(self):
  #   S = self.corr.symmetrize(overwrite=False, method='sum')
  #   expected = self.m + self.m.T
  #   assert_array_equal(S.matrix().A, expected)
  #   self.assertTrue(S is not self.corr)

  #   S = self.corr.symmetrize(overwrite=False, method='avg')
  #   expected = (self.m + self.m.T) / 2.0
  #   assert_array_equal(S.matrix().A, expected)

  #   S = self.corr.symmetrize(overwrite=False, method='max')
  #   expected = numpy.maximum(self.m, self.m.T)
  #   assert_array_equal(S.matrix().A, expected)

  #   # Check the overwrite=True behavior
  #   C = graph.Graph(matrix=self.corr.matrix().copy())
  #   S = C.symmetrize()
  #   assert_array_equal(S.matrix().A, expected)
  #   self.assertTrue(S is C)
