import matplotlib
matplotlib.use('template')

import unittest
import numpy as np
import warnings
from numpy.testing import assert_array_equal
from scipy.sparse import csr_matrix

from graphs.base import (
    EdgePairGraph, DenseAdjacencyMatrixGraph, SparseAdjacencyMatrixGraph
)

PAIRS = np.array([[0,1],[0,2],[1,1],[2,1],[3,3]])
ADJ = [[0,1,1,0],
       [0,1,0,0],
       [0,1,0,0],
       [0,0,0,1]]


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

  def test_add_self_edges_unweighted(self):
    expected = (np.array(ADJ) + np.eye(len(ADJ))).astype(bool).astype(int)
    for G in self.graphs:
      gg = G.add_self_edges()
      self.assertIs(gg, G)
      self.assertEqual(G.num_edges(), 7)
      assert_array_equal(G.matrix(dense=True), expected,
                         'unweighted (%s)' % type(G))
    with warnings.catch_warnings(record=True) as w:
      self.graphs[0].add_self_edges(weight=3)
      self.assertEqual(len(w), 1)
      self.assertIn('ignoring weight argument', w[0].message.message)

  def test_add_self_edges_weighted(self):
    wg = [G for G in self.graphs if G.is_weighted()]
    expected = np.array(ADJ, dtype=float)
    np.fill_diagonal(expected, 0.5)
    for G in wg:
      G.add_self_edges(weight=0.5)
      self.assertEqual(G.num_edges(), 7)
      assert_array_equal(G.matrix(dense=True), expected,
                         'weighted (%s)' % type(G))
    # zeros case
    np.fill_diagonal(expected, 0)
    for G in wg:
      G.add_self_edges(weight=0)
      self.assertEqual(G.num_edges(), 3)
      assert_array_equal(G.matrix(dense=True), expected,
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
        self.assertIsNot(G.edge_weights(copy=True), ew)
    expected = [1,2,1,1,3]
    assert_array_equal(self.weighted.edge_weights(), expected)

  def test_add_edges_unweighted(self):
    expected = np.array(ADJ)
    from_idx = [0,3,2]
    to_idx = [2,2,2]
    expected[from_idx,to_idx] = 1
    for G in self.graphs:
      msg = 'unweighted (%s)' % type(G)
      gg = G.add_edges(from_idx, to_idx)
      self.assertIs(gg, G)
      self.assertEqual(G.num_edges(), 7, msg)
      assert_array_equal(G.matrix(dense=True), expected, msg)
    # symmetric version
    expected[to_idx,from_idx] = 1
    for G in self.graphs:
      msg = 'unweighted symmetric (%s)' % type(G)
      gg = G.add_edges(from_idx, to_idx, symmetric=True)
      self.assertIs(gg, G)
      self.assertEqual(G.num_edges(), 9, msg)
      assert_array_equal(G.matrix(dense=True), expected, msg)

  def test_add_edges_weighted(self):
    wg = [G for G in self.graphs if G.is_weighted()]
    expected = np.array(ADJ, dtype=float)
    from_idx = [0,3,2]
    to_idx = [2,2,2]
    expected[from_idx,to_idx] = 1
    for G in wg:
      msg = 'weighted (%s)' % type(G)
      gg = G.add_edges(from_idx, to_idx, weight=1)
      self.assertIs(gg, G)
      self.assertEqual(G.num_edges(), 7, msg)
      assert_array_equal(G.matrix(dense=True), expected, msg)
    # symmetric version
    expected[to_idx,from_idx] = 1
    for G in wg:
      msg = 'weighted symmetric (%s)' % type(G)
      gg = G.add_edges(from_idx, to_idx, weight=1, symmetric=True)
      self.assertIs(gg, G)
      self.assertEqual(G.num_edges(), 9, msg)
      assert_array_equal(G.matrix(dense=True), expected, msg)

  def test_add_edges_zeros(self):
    wg = [G for G in self.graphs if G.is_weighted()]
    expected = np.array(ADJ, dtype=float)
    from_idx = [0,3,2]
    to_idx = [2,2,2]
    expected[from_idx,to_idx] = 0
    for G in wg:
      msg = 'zero-weight (%s)' % type(G)
      gg = G.add_edges(from_idx, to_idx, weight=0)
      self.assertIs(gg, G)
      self.assertEqual(G.num_edges(), 4, msg)
      assert_array_equal(G.matrix(dense=True), expected, msg)

  def test_add_edges_array_weighted(self):
    wg = [G for G in self.graphs if G.is_weighted()]
    weights = np.linspace(1, 9, 3)
    expected = np.array(ADJ, dtype=float)
    from_idx = [0,3,2]
    to_idx = [2,2,2]
    expected[from_idx,to_idx] = weights
    for G in wg:
      msg = 'array-weighted (%s)' % type(G)
      gg = G.add_edges(from_idx, to_idx, weight=weights)
      self.assertIs(gg, G)
      self.assertEqual(G.num_edges(), 7, msg)
      assert_array_equal(G.matrix(dense=True), expected, msg)
    # symmetric version
    expected[to_idx,from_idx] = weights
    for G in wg:
      msg = 'array-weighted symmetric (%s)' % type(G)
      gg = G.add_edges(from_idx, to_idx, weight=weights, symmetric=True)
      self.assertIs(gg, G)
      self.assertEqual(G.num_edges(), 9, msg)
      assert_array_equal(G.matrix(dense=True), expected, msg)

  def test_to_igraph(self):
    try:
      for G in self.graphs + [self.weighted]:
        ig = G.to_igraph()
        if G.is_weighted():
          adj = ig.get_adjacency(attribute='weight')
        else:
          adj = ig.get_adjacency()
        assert_array_equal(G.matrix(dense=True), adj.data)
    except ImportError:
      return  # Skip this test.

  def test_to_graph_tool(self):
    try:
      from graph_tool.spectral import adjacency
    except ImportError:
      return  # Skip this test.
    for G in self.graphs + [self.weighted]:
      gt = G.to_graph_tool()
      if G.is_weighted():
        adj = adjacency(gt, weight=gt.ep['weight']).A.T
      else:
        adj = adjacency(gt).A.T
      assert_array_equal(G.matrix(dense=True), adj)

  def test_reweight(self):
    wg = [G for G in self.graphs if G.is_weighted()]
    expected = np.array(ADJ, dtype=float)
    mask = expected != 0
    new_weights = np.arange(1, np.count_nonzero(mask)+1)
    expected[mask] = new_weights
    for G in wg:
      msg = 'reweight (%s)' % type(G)
      gg = G.reweight(new_weights)
      self.assertIs(gg, G)
      self.assertEqual(G.num_edges(), 5, msg)
      assert_array_equal(G.matrix(dense=True), expected, msg)

  def test_reweight_partial(self):
    wg = [G for G in self.graphs if G.is_weighted()]
    expected = np.array(ADJ, dtype=float)
    ii, jj = np.where(expected)
    new_weight_inds = [2,3]
    new_weights = np.array([5,6])
    expected[ii[new_weight_inds], jj[new_weight_inds]] = new_weights
    for G in wg:
      msg = 'reweight partial (%s)' % type(G)
      gg = G.reweight(new_weights, new_weight_inds)
      self.assertIs(gg, G)
      self.assertEqual(G.num_edges(), 5, msg)
      assert_array_equal(G.matrix(dense=True), expected, msg)

if __name__ == '__main__':
  unittest.main()
