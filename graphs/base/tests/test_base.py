import unittest
import numpy as np
import warnings
from numpy.testing import assert_array_equal
from scipy.sparse import lil_matrix

from graphs.base import (
    EdgePairGraph, SymmEdgePairGraph,
    DenseAdjacencyMatrixGraph, SparseAdjacencyMatrixGraph
)

try:
  import igraph
  HAS_IGRAPH = True
except ImportError:
  HAS_IGRAPH = False

try:
  import graph_tool
  HAS_GRAPHTOOL = True
except ImportError:
  HAS_GRAPHTOOL = False

try:
  import networkx
  HAS_NETWORKX = True
except ImportError:
  HAS_NETWORKX = False

PAIRS = np.array([[0,1],[0,2],[1,1],[2,1],[3,3]])
ADJ = [[0,1,1,0],
       [0,1,0,0],
       [0,1,0,0],
       [0,0,0,1]]


class TestGenericMembers(unittest.TestCase):
  def setUp(self):
    spadj = lil_matrix(ADJ)
    spadj[0,0] = 0  # Add an explicit zero
    self.graphs = [
        EdgePairGraph(PAIRS),
        DenseAdjacencyMatrixGraph(ADJ),
        SparseAdjacencyMatrixGraph(spadj),
        SparseAdjacencyMatrixGraph(spadj.tocoo())
    ]
    self.weighted = DenseAdjacencyMatrixGraph(np.array(ADJ)*np.arange(4)[None])
    self.sym = SymmEdgePairGraph(PAIRS.copy(), num_vertices=4)

  def test_properties(self):
    for G in self.graphs:
      self.assertEqual(G.num_edges(), 5, 'num_edges (%s)' % type(G))
      self.assertEqual(G.num_vertices(), 4, 'num_vertices (%s)' % type(G))

  def test_copy(self):
    for G in self.graphs:
      gg = G.copy()
      self.assertIsNot(gg, G)
      assert_array_equal(gg.matrix(dense=True), G.matrix(dense=True))
      assert_array_equal(gg.pairs(), G.pairs())

  def test_degree(self):
    for G in self.graphs:
      in_degree = G.degree('in', weighted=False)
      out_degree = G.degree('out', weighted=False)
      assert_array_equal(in_degree, [0, 3, 1, 1])
      assert_array_equal(out_degree, [2, 1, 1, 1])

  def test_degree_weighted(self):
    in_degree = self.weighted.degree(kind='in', weighted=True)
    out_degree = self.weighted.degree(kind='out', weighted=True)
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
      self.assertEqual(G.num_edges(), np.count_nonzero(expected))
      assert_array_equal(G.matrix(dense=True), expected,
                         'unweighted (%s)' % type(G))
    with warnings.catch_warnings(record=True) as w:
      self.graphs[0].add_self_edges(weight=3)
      self.assertEqual(len(w), 1)
      self.assertIn('ignoring weight argument', str(w[0].message))

  def test_add_self_edges_weighted(self):
    wg = [G for G in self.graphs if G.is_weighted()]
    expected = np.array(ADJ, dtype=float)
    np.fill_diagonal(expected, 0.5)
    for G in wg:
      G.add_self_edges(weight=0.5)
      self.assertEqual(G.num_edges(), np.count_nonzero(expected))
      assert_array_equal(G.matrix(dense=True), expected,
                         'weighted (%s)' % type(G))
    # zeros case
    np.fill_diagonal(expected, 0)
    for G in wg:
      G.add_self_edges(weight=0)
      self.assertEqual(G.num_edges(), np.count_nonzero(expected))
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
      sym = G.symmetrize(method=method, copy=True).matrix(dense=True)
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
    from_idx = [2,3,0]
    to_idx = [2,2,2]
    expected[from_idx,to_idx] = 1
    for G in self.graphs:
      msg = 'unweighted (%s)' % type(G)
      g1 = G.add_edges(from_idx, to_idx, copy=True)
      self.assertIsNot(g1, G)
      g2 = G.add_edges(from_idx, to_idx)
      self.assertIs(g2, G)
      self.assertEqual(G.num_edges(), np.count_nonzero(expected), msg)
      assert_array_equal(G.matrix(dense=True), expected, msg)
      assert_array_equal(g1.matrix(dense=True), expected, msg)
    # symmetric version
    expected[to_idx,from_idx] = 1
    for G in self.graphs:
      msg = 'unweighted symmetric (%s)' % type(G)
      gg = G.add_edges(from_idx, to_idx, symmetric=True)
      self.assertIs(gg, G)
      self.assertEqual(G.num_edges(), np.count_nonzero(expected))
      assert_array_equal(G.matrix(dense=True), expected)

  def test_add_edges_weighted(self):
    wg = [G for G in self.graphs if G.is_weighted()]
    expected = np.array(ADJ, dtype=float)
    from_idx = [2,3,0]
    to_idx = [2,2,2]
    expected[from_idx,to_idx] = 1
    for G in wg:
      msg = 'weighted (%s)' % type(G)
      gg = G.add_edges(from_idx, to_idx, weight=1)
      self.assertIs(gg, G)
      self.assertEqual(G.num_edges(), np.count_nonzero(expected), msg)
      assert_array_equal(G.matrix(dense=True), expected, msg)
    # symmetric version
    expected[to_idx,from_idx] = 1
    for G in wg:
      msg = 'weighted symmetric (%s)' % type(G)
      gg = G.add_edges(from_idx, to_idx, weight=1, symmetric=True)
      self.assertIs(gg, G)
      self.assertEqual(G.num_edges(), np.count_nonzero(expected), msg)
      assert_array_equal(G.matrix(dense=True), expected, msg)

  def test_add_edges_zeros(self):
    wg = [G for G in self.graphs if G.is_weighted()]
    expected = np.array(ADJ, dtype=float)
    from_idx = [2,3,0]
    to_idx = [2,2,2]
    expected[from_idx,to_idx] = 0
    for G in wg:
      msg = 'zero-weight (%s)' % type(G)
      gg = G.add_edges(from_idx, to_idx, weight=0)
      self.assertIs(gg, G)
      self.assertEqual(G.num_edges(), np.count_nonzero(expected), msg)
      assert_array_equal(G.matrix(dense=True), expected, msg)

  def test_add_edges_array_weighted(self):
    wg = [G for G in self.graphs if G.is_weighted()]
    weights = np.linspace(1, 9, 3)
    expected = np.array(ADJ, dtype=float)
    from_idx = [2,3,0]
    to_idx = [2,2,2]
    expected[from_idx,to_idx] = weights
    for G in wg:
      msg = 'array-weighted (%s)' % type(G)
      gg = G.add_edges(from_idx, to_idx, weight=weights)
      self.assertIs(gg, G)
      self.assertEqual(G.num_edges(), np.count_nonzero(expected), msg)
      assert_array_equal(G.matrix(dense=True), expected, msg)
    # symmetric version
    expected[to_idx,from_idx] = weights
    for G in wg:
      msg = 'array-weighted symmetric (%s)' % type(G)
      gg = G.add_edges(from_idx, to_idx, weight=weights, symmetric=True)
      self.assertIs(gg, G)
      self.assertEqual(G.num_edges(), np.count_nonzero(expected), msg)
      assert_array_equal(G.matrix(dense=True), expected, msg)

  @unittest.skipUnless(HAS_IGRAPH, 'requires igraph dependency')
  def test_to_igraph(self):
    for G in self.graphs + [self.weighted]:
      ig = G.to_igraph()
      if G.is_weighted():
        adj = ig.get_adjacency(attribute='weight')
      else:
        adj = ig.get_adjacency()
      assert_array_equal(G.matrix(dense=True), adj.data)

  @unittest.skipUnless(HAS_GRAPHTOOL, 'requires graph_tool dependency')
  def test_to_graph_tool(self):
    from graph_tool.spectral import adjacency
    for G in self.graphs + [self.weighted]:
      gt = G.to_graph_tool()
      if G.is_weighted():
        adj = adjacency(gt, weight=gt.ep['weight']).A.T
      else:
        adj = adjacency(gt).A.T
      assert_array_equal(G.matrix(dense=True), adj)

  @unittest.skipUnless(HAS_NETWORKX, 'requires networkx dependency')
  def test_to_networkx(self):
    for G in self.graphs + [self.weighted]:
      nx = G.to_networkx()
      adj = networkx.to_numpy_matrix(nx)
      assert_array_equal(G.matrix(dense=True), adj)

  def test_reweight(self):
    expected = np.array(ADJ, dtype=float)
    mask = expected != 0
    new_weights = np.arange(1, np.count_nonzero(mask)+1)
    expected[mask] = new_weights
    for G in self.graphs:
      if G.is_weighted():
        msg = 'reweight (%s)' % type(G)
        gg = G.reweight(new_weights)
        self.assertIs(gg, G)
        self.assertEqual(G.num_edges(), np.count_nonzero(expected), msg)
        assert_array_equal(G.matrix(dense=True), expected, msg)
      else:
        with warnings.catch_warnings(record=True) as w:
          G.reweight(new_weights)
          self.assertEqual(len(w), 1)
          self.assertIn('ignoring call to reweight', str(w[0].message))

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
      self.assertEqual(G.num_edges(), np.count_nonzero(expected), msg)
      assert_array_equal(G.matrix(dense=True), expected, msg)

  def test_reweight_by_distance(self):
    wg = [G for G in self.graphs if G.is_weighted()]
    expected = np.array(ADJ, dtype=float)
    mask = expected != 0
    coords = np.arange(np.count_nonzero(mask))[:,None]
    expected[mask] = np.abs(PAIRS[:,0] - PAIRS[:,1])
    for G in wg:
      msg = 'reweight_by_distance (%s)' % type(G)
      gg = G.reweight_by_distance(coords, metric='l2')
      self.assertIs(gg, G)
      self.assertEqual(G.num_edges(), np.count_nonzero(expected), msg)
      assert_array_equal(G.matrix(dense=True), expected, msg)

  def test_remove_edges(self):
    for G in self.graphs:
      gg = G.remove_edges(0, 2, copy=True)
      assert_array_equal(gg.pairs(), [[0,1],[1,1],[2,1],[3,3]])
      gg = G.remove_edges([0,1], [2,2], symmetric=True, copy=True)
      assert_array_equal(gg.pairs(), [[0,1],[1,1],[3,3]])
      # make sure we didn't modify G
      assert_array_equal(G.pairs(), PAIRS)
      # now actually modify G
      gg = G.remove_edges(0, 2)
      self.assertIs(gg, G)
      assert_array_equal(G.pairs(), [[0,1],[1,1],[2,1],[3,3]])

    gg = self.sym.remove_edges([0,1], [2,2], copy=True)
    assert_array_equal(gg.pairs(), [[0,1],[1,0],[1,1],[3,3]])

  def test_subgraph(self):
    adj = np.array(ADJ, dtype=float)
    for G in self.graphs:
      # entire graph in the subgraph
      gg = G.subgraph(Ellipsis)
      self.assertEqual(type(gg), type(G))
      assert_array_equal(gg.matrix(dense=True), adj)

      # half the graph
      mask = slice(0, 2)
      gg = G.subgraph(mask)
      self.assertEqual(type(gg), type(G))
      assert_array_equal(gg.matrix(dense=True), adj[mask][:,mask])

      mask = np.array([False, True, True, False])
      gg = G.subgraph(mask)
      self.assertEqual(type(gg), type(G))
      assert_array_equal(gg.matrix(dense=True), adj[mask][:,mask])


if __name__ == '__main__':
  unittest.main()
