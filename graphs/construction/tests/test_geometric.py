import matplotlib
matplotlib.use('template')

import numpy as np
import unittest
from numpy.testing import assert_array_equal

from graphs.construction import (
    delaunay_graph, gabriel_graph, relative_neighborhood_graph)


class TestGeometric(unittest.TestCase):
  def setUp(self):
    self.pts = np.array([[0,0],[1.1,2],[3.3,2],[-0.9,0],[0.5,1]])

  def test_delaunay(self):
    expected = [[0,0,1,1,1],[0,0,1,1,1],[1,1,0,0,1],[1,1,0,0,1],[1,1,1,1,0]]
    G = delaunay_graph(self.pts)
    assert_array_equal(G.matrix(dense=True), expected)

  def test_gabriel(self):
    expected = [[0,0,0,1,1],[0,0,1,0,1],[0,1,0,0,0],[1,0,0,0,0],[1,1,0,0,0]]
    G = gabriel_graph(self.pts)
    assert_array_equal(G.matrix(dense=True), expected)

  def test_relative_neighborhood(self):
    expected = [[0,0,0,1,1],[0,0,1,0,1],[0,1,0,0,0],[1,0,0,0,0],[1,1,0,0,0]]
    G = relative_neighborhood_graph(self.pts)
    assert_array_equal(G.matrix(dense=True), expected)

if __name__ == '__main__':
  unittest.main()
