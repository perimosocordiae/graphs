import unittest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from scipy.sparse import coo_matrix, csr_matrix
from graphs import Graph
from graphs.construction import neighbor_graph

PAIRS = np.array([[0,1],[0,2],[1,0],[1,2],[2,0],[2,1],[3,4],[4,3]])
ADJ = [[0,1,1,0,0],
       [1,0,1,0,0],
       [1,1,0,0,0],
       [0,0,0,0,1],
       [0,0,0,1,0]]

# fixed "random" data in 2 dimensions
X = np.column_stack((
    [0.192, 0.438, 0.78, 0.276, 0.958, 0.358, 0.683, 0.37, 0.503, 0.773,
     0.365, 0.075, 0.933, 0.397, 0.317, 0.869, 0.802, 0.704, 0.219, 0.442],
    [0.622, 0.785, 0.273, 0.802, 0.876, 0.501, 0.713, 0.561, 0.014, 0.883,
     0.615, 0.369, 0.651, 0.789, 0.568, 0.436, 0.144, 0.705, 0.925, 0.909]
))


class TestTransformation(unittest.TestCase):

  def test_kernelize(self):
    graphs = [
        Graph.from_edge_pairs(PAIRS),
        Graph.from_adj_matrix(ADJ),
        Graph.from_adj_matrix(coo_matrix(ADJ)),
        Graph.from_adj_matrix(csr_matrix(ADJ)),
    ]
    for G in graphs:
      for kernel in ('none', 'binary'):
        K = G.kernelize(kernel)
        assert_array_equal(K.matrix(dense=True), ADJ)
      self.assertRaises(ValueError, G.kernelize, 'foobar')

  def test_connected_subgraphs(self):
    G = Graph.from_edge_pairs(PAIRS)
    subgraphs = list(G.connected_subgraphs(directed=False, ordered=False))
    self.assertEqual(len(subgraphs), 2)
    assert_array_equal(subgraphs[0].pairs(), PAIRS[:6])
    assert_array_equal(subgraphs[1].pairs(), [[0,1],[1,0]])

    G = neighbor_graph(X, k=2)
    subgraphs = list(G.connected_subgraphs(directed=True, ordered=True))
    self.assertEqual(len(subgraphs), 3)
    self.assertEqual([g.num_vertices() for g in subgraphs], [9,6,5])

  def test_shortest_path_subtree(self):
    n = X.shape[0]
    G = neighbor_graph(X, k=4)
    e_data = [0.163, 0.199, 0.079, 0.188, 0.173, 0.122, 0.136, 0.136, 0.197]
    e_row = [3, 0, 14, 0, 0, 3, 0, 3, 3]
    e_col = [1, 3, 5, 7, 10, 13, 14, 18, 19]
    expected = np.zeros((n,n))
    expected[e_row, e_col] = e_data

    spt = G.shortest_path_subtree(0, directed=True)
    assert_array_almost_equal(spt.matrix(dense=True), expected, decimal=3)

    # test undirected case
    G.symmetrize(method='max', copy=False)
    e_data = [0.185,0.379,0.199,0.32,0.205,0.255,0.188,0.508,0.192,0.173,0.279,
              0.258,0.122,0.136,0.316,0.326,0.278,0.136,0.197,0.185,0.379,0.199,
              0.32,0.205,0.255,0.188,0.508,0.192,0.173,0.279,0.258,0.122,0.136,
              0.316,0.326,0.278,0.136,0.197]
    e_row = [10,8,0,6,0,1,0,5,6,0,0,6,3,0,17,8,1,3,3,1,2,3,4,5,6,7,8,9,10,11,12,
             13,14,15,16,17,18,19]
    e_col = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,10,8,0,6,0,1,0,5,6,
             0,0,6,3,0,17,8,1,3,3]
    expected[:] = 0
    expected[e_row, e_col] = e_data

    spt = G.shortest_path_subtree(0, directed=False)
    assert_array_almost_equal(spt.matrix(dense=True), expected, decimal=3)

  def test_minimum_spanning_subtree(self):
    n = X.shape[0]
    G = neighbor_graph(X, k=4)
    e_data = [0.279,0.136,0.255,0.041,0.124,0.186,0.131,0.122,0.136,0.185,0.226,
              0.061,0.255,0.022,0.061,0.054,0.053,0.326,0.185,0.191,0.054,0.177,
              0.279,0.226,0.224,0.041,0.122,0.177,0.136,0.053,0.186,0.224,0.131,
              0.326,0.022,0.191,0.136,0.124]
    e_row = [0,0,1,1,1,2,2,3,3,4,4,5,6,6,7,7,7,8,9,9,10,10,11,12,12,13,13,13,14,
             14,15,15,16,16,17,17,18,19]
    e_col = [11,14,6,13,19,15,16,13,18,9,12,7,1,17,5,10,14,16,4,17,7,13,0,4,15,
             1,3,10,0,7,2,12,2,8,6,9,3,1]
    expected = np.zeros((n,n))
    expected[e_row, e_col] = e_data

    mst = G.minimum_spanning_subtree()
    assert_array_almost_equal(mst.matrix(dense=True), expected, decimal=3)

  def test_circle_tear(self):
    G = neighbor_graph(X, k=4).symmetrize(method='max', copy=False)

    # test MST start
    res = G.circle_tear(spanning_tree='mst', cycle_len_thresh=5)
    diff = G.matrix(dense=True) - res.matrix(dense=True)
    ii, jj = np.nonzero(diff)
    assert_array_equal(ii, [5,8,8,11])
    assert_array_equal(jj, [8,5,11,8])

    # test SPT start with a fixed starting vertex
    res = G.circle_tear(spanning_tree='spt', cycle_len_thresh=5, spt_idx=8)
    diff = G.matrix(dense=True) - res.matrix(dense=True)
    ii, jj = np.nonzero(diff)
    assert_array_equal(ii, [1,1,6,17])
    assert_array_equal(jj, [6,17,1,1])

  def test_cycle_cut(self):
    G = neighbor_graph(X, k=4).symmetrize(method='max', copy=False)

    # hack: the atomic cycle finder chooses a random vertex to start from
    np.random.seed(1234)
    res = G.cycle_cut(cycle_len_thresh=5, directed=False)
    diff = G.matrix(dense=True) - res.matrix(dense=True)
    ii, jj = np.nonzero(diff)
    assert_array_equal(ii, [1,1,6,17])
    assert_array_equal(jj, [6,17,1,1])

if __name__ == '__main__':
  unittest.main()
