import unittest
import numpy as np
from numpy.testing import assert_array_equal
from scipy.sparse import csr_matrix

from graphs.base import Graph, EdgePairGraph

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
