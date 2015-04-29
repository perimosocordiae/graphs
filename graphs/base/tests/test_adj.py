import unittest
import numpy as np
from numpy.testing import assert_array_equal
from scipy.sparse import csr_matrix

from graphs.base.adj import (
    DenseAdjacencyMatrixGraph, SparseAdjacencyMatrixGraph)

PAIRS = np.array([[0,1],[0,2],[1,1],[2,1],[3,3]])
ADJ = [[0,1,1,0],
       [0,1,0,0],
       [0,1,0,0],
       [0,0,0,1]]


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

  def test_matrix_copy(self):
    M = self.G.matrix(dense=True, copy=False)
    assert_array_equal(M, ADJ)
    M2 = self.G.matrix(dense=True, copy=True)
    assert_array_equal(M, M2)
    self.assertIsNot(M, M2)
    # Sparse case
    M = self.S.matrix(csr=True, copy=False)
    assert_array_equal(M.toarray(), ADJ)
    M2 = self.S.matrix(csr=True, copy=True)
    assert_array_equal(M.toarray(), M2.toarray())
    self.assertIsNot(M, M2)

if __name__ == '__main__':
  unittest.main()
