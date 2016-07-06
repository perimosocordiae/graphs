import numpy as np
import unittest
from numpy.testing import assert_array_equal, assert_array_almost_equal
from sklearn.metrics import pairwise_distances

from graphs.construction import (
    delaunay_graph, gabriel_graph, relative_neighborhood_graph)
from graphs.construction.geometric import _find_relative_neighbors


class TestGeometric(unittest.TestCase):
  def setUp(self):
    self.pts = np.array([
        [0.192,0.622],[0.438,0.785],[0.780,0.273],[0.276,0.802],[0.958,0.876],
        [0.358,0.501],[0.683,0.713],[0.370,0.561],[0.503,0.014],[0.773,0.883]])

  def test_delaunay(self):
    expected = [
        [0, 0, 0, 1, 0, 1, 0, 1, 1, 0],
        [0, 0, 0, 1, 0, 0, 1, 1, 0, 1],
        [0, 0, 0, 0, 1, 1, 1, 0, 1, 0],
        [1, 1, 0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
        [1, 0, 1, 0, 0, 0, 1, 1, 1, 0],
        [0, 1, 1, 0, 1, 1, 0, 1, 0, 1],
        [1, 1, 0, 1, 0, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 1, 0, 1, 0, 0, 0]]
    G = delaunay_graph(self.pts)
    assert_array_equal(G.matrix(dense=True), expected)

    # with edge weights
    G = delaunay_graph(self.pts, weighted=True)
    expected = [
        0.198635, 0.205419, 0.188162, 0.682924, 0.16289, 0.255361,
        0.234094, 0.34904, 0.628723, 0.479654, 0.450565, 0.379223,
        0.198635, 0.16289, 0.258683, 0.503557, 0.628723, 0.319678,
        0.185132, 0.205419, 0.479654, 0.388032, 0.061188, 0.508128,
        0.255361, 0.450565, 0.319678, 0.388032, 0.347955, 0.192354,
        0.188162, 0.234094, 0.258683, 0.061188, 0.347955, 0.682924,
        0.379223, 0.508128, 0.34904, 0.503557, 0.185132, 0.192354]
    assert_array_almost_equal(G.edge_weights(), expected)

  def test_gabriel(self):
    expected = np.array([
        [0,3], [0,7], [1,3], [1,6], [1,7], [2,5], [2,6], [2,8], [3,7], [4,9],
        [5,7], [5,8], [6,9]])
    expected = np.vstack((expected, expected[:,::-1]))
    G = gabriel_graph(self.pts)
    assert_array_equal(G.pairs(), expected)

    # with edge weights
    G = gabriel_graph(self.pts, weighted=True)
    expected = [
        0.198635, 0.188162, 0.16289, 0.255361, 0.234094, 0.479654,
        0.450565, 0.379223, 0.198635, 0.16289, 0.258683, 0.185132,
        0.479654, 0.061188, 0.508128, 0.255361, 0.450565, 0.192354,
        0.188162, 0.234094, 0.258683, 0.061188, 0.379223, 0.508128,
        0.185132, 0.192354]
    assert_array_almost_equal(G.edge_weights(), expected)

  def test_relative_neighborhood(self):
    dist = pairwise_distances(self.pts)
    expected = np.array([
        [0,3], [0,7], [1,3], [1,6], [1,7], [2,6], [2,8], [4,9], [5,7], [6,9]])

    pairs = np.asarray(_find_relative_neighbors(dist))
    assert_array_equal(pairs, expected)

    expected = np.vstack((expected, expected[:,::-1]))
    G = relative_neighborhood_graph(self.pts)
    assert_array_equal(G.pairs(), expected)

    # with metric='precomputed'
    G = relative_neighborhood_graph(dist, metric='precomputed')
    assert_array_equal(G.pairs(), expected)

    # with edge weights
    G = relative_neighborhood_graph(self.pts, weighted=True)
    expected = [
        0.198635, 0.188162, 0.16289, 0.255361, 0.234094, 0.450565,
        0.379223, 0.198635, 0.16289, 0.185132, 0.061188, 0.255361,
        0.450565, 0.192354, 0.188162, 0.234094, 0.061188, 0.379223,
        0.185132, 0.192354]
    assert_array_almost_equal(G.edge_weights(), expected)

if __name__ == '__main__':
  unittest.main()
