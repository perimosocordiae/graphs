import unittest
import numpy as np
from numpy.testing import assert_array_equal
from scipy.sparse import coo_matrix, csr_matrix
from graphs import Graph

PAIRS = np.array([[0,1],[0,2],[1,0],[1,2],[2,0],[2,1],[3,4],[4,3]])
ADJ = [[0,1,1,0,0],
       [1,0,1,0,0],
       [1,1,0,0,0],
       [0,0,0,0,1],
       [0,0,0,1,0]]


class TestTransformation(unittest.TestCase):
  def setUp(self):
    self.graphs = [
        Graph.from_edge_pairs(PAIRS),
        Graph.from_adj_matrix(ADJ),
        Graph.from_adj_matrix(coo_matrix(ADJ)),
        Graph.from_adj_matrix(csr_matrix(ADJ)),
    ]

  def test_kernelize(self):
    for G in self.graphs:
      for kernel in ('none', 'binary'):
        K = G.kernelize(kernel)
        assert_array_equal(K.matrix(dense=True), ADJ)
      self.assertRaises(ValueError, G.kernelize, 'foobar')

if __name__ == '__main__':
  unittest.main()
