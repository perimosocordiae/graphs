import numpy as np
import unittest
from numpy.testing import assert_array_equal

from graphs.construction import (
    delaunay_graph, gabriel_graph, relative_neighborhood_graph)


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

  def test_gabriel(self):
    expected = np.array([
        [0,3], [0,7], [1,3], [1,6], [1,7], [2,5], [2,6], [2,8], [3,7], [4,9],
        [5,7], [5,8], [6,9]])
    expected = np.vstack((expected, expected[:,::-1]))
    G = gabriel_graph(self.pts)
    assert_array_equal(G.pairs(), expected)

  def test_relative_neighborhood(self):
    expected = np.array([
        [0,3], [0,7], [1,3], [1,6], [1,7], [2,6], [2,8], [4,9], [5,7], [6,9]])
    expected = np.vstack((expected, expected[:,::-1]))
    G = relative_neighborhood_graph(self.pts)
    assert_array_equal(G.pairs(), expected)

if __name__ == '__main__':
  unittest.main()
