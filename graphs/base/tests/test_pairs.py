import matplotlib
matplotlib.use('template')

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

  def test_self_edges(self):
    self.epg.add_self_edges()
    expected = self.epg.pairs()
    # Ensure that calling it again does the right thing.
    self.epg.add_self_edges()
    assert_array_equal(self.epg.pairs(), expected)

  def test_symmetrize(self):
    # Check that overwrite=False doesn't change anything
    self.epg.symmetrize(overwrite=False)
    assert_array_equal(self.epg.matrix(dense=True), ADJ)


class TestSymmEdgePairGraph(unittest.TestCase):
  def setUp(self):
    self.G = SymmEdgePairGraph(PAIRS)

  def test_pairs(self):
    expected = [[0,1], [0,2], [1,0], [1,1], [1,2], [2,0], [2,1], [3,3]]
    P = self.G.pairs()
    assert_array_equal(sorted(P.tolist()), expected)

  def test_symmetrize(self):
    self.assertIs(self.G.symmetrize(overwrite=True), self.G)
    S = self.G.symmetrize(overwrite=False)
    self.assertIsNot(S, self.G)
    assert_array_equal(S.matrix(dense=True), self.G.matrix(dense=True))

if __name__ == '__main__':
  unittest.main()