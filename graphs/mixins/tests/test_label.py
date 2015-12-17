import unittest
import numpy as np
from numpy.testing import assert_array_equal
from scipy.sparse import coo_matrix
from graphs import Graph

PAIRS = np.array([[0,1],[0,2],[1,0],[1,2],[2,0],[2,1],[3,4],[4,3]])
ADJ = [[0,1,1,0,0],
       [1,0,1,0,0],
       [1,1,0,0,0],
       [0,0,0,0,1],
       [0,0,0,1,0]]


class TestLabel(unittest.TestCase):
  def setUp(self):
    self.graphs = [
        Graph.from_edge_pairs(PAIRS),
        Graph.from_adj_matrix(ADJ),
        Graph.from_adj_matrix(coo_matrix(ADJ)),
    ]

  def test_greedy_coloring(self):
    for G in self.graphs:
      assert_array_equal([1,2,3,1,2], G.greedy_coloring())


if __name__ == '__main__':
  unittest.main()
