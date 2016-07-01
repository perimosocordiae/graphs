import unittest
import numpy as np
from numpy.testing import assert_array_equal

from graphs.base.pairs import EdgePairGraph, SymmEdgePairGraph

PAIRS = np.array([[0,1],[0,2],[1,1],[2,1],[3,3]])
ADJ = [[0,1,1,0],
       [0,1,0,0],
       [0,1,0,0],
       [0,0,0,1]]


class TestEdgePairGraph(unittest.TestCase):
  def setUp(self):
    self.epg = EdgePairGraph(PAIRS)

  def test_pairs(self):
    self.assert_(self.epg.pairs(copy=False) is PAIRS)
    P = self.epg.pairs(copy=True)
    self.assert_(P is not PAIRS)
    assert_array_equal(P, PAIRS)
    # test the directed case
    P = self.epg.pairs(directed=False)
    assert_array_equal(P, [[0,1],[0,2],[1,1],[1,2],[3,3]])

  def test_matrix(self):
    M = self.epg.matrix()
    assert_array_equal(M.toarray(), ADJ)
    M = self.epg.matrix(dense=True)
    assert_array_equal(M, ADJ)
    M = self.epg.matrix(csr=True)
    self.assertEqual(M.format, 'csr')
    assert_array_equal(M.toarray(), ADJ)

  def test_self_edges(self):
    self.epg.add_self_edges()
    expected = self.epg.pairs()
    # Ensure that calling it again does the right thing.
    self.epg.add_self_edges()
    assert_array_equal(self.epg.pairs(), expected)

  def test_symmetrize(self):
    # Check that copy=True doesn't change anything
    self.epg.symmetrize(copy=True)
    assert_array_equal(self.epg.matrix(dense=True), ADJ)


class TestSymmEdgePairGraph(unittest.TestCase):
  def setUp(self):
    self.G = SymmEdgePairGraph(PAIRS)

  def test_copy(self):
    gg = self.G.copy()
    self.assertIsNot(gg, self.G)
    assert_array_equal(gg.matrix(dense=True), self.G.matrix(dense=True))
    assert_array_equal(gg.pairs(), self.G.pairs())

  def test_pairs(self):
    expected = [[0,1], [0,2], [1,0], [1,1], [1,2], [2,0], [2,1], [3,3]]
    P = self.G.pairs()
    assert_array_equal(sorted(P.tolist()), expected)
    # test the directed case
    P = self.G.pairs(directed=False)
    assert_array_equal(P, [[0,1],[0,2],[1,1],[1,2],[3,3]])

  def test_symmetrize(self):
    self.assertIs(self.G.symmetrize(copy=False), self.G)
    S = self.G.symmetrize(copy=True)
    self.assertIsNot(S, self.G)
    assert_array_equal(S.matrix(dense=True), self.G.matrix(dense=True))

if __name__ == '__main__':
  unittest.main()
