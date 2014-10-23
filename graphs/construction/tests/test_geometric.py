import matplotlib
matplotlib.use('template')

import numpy as np
import unittest
from numpy.testing import assert_array_equal

from graphs.construction import (
    delaunay_graph, gabriel_graph, relative_neighborhood_graph)


class TestGeometric(unittest.TestCase):

  def test_delaunay(self):
    pts = np.array([[0,0],[1,2],[3,2],[-1,0]])

  def test_gabriel(self):
    pts = np.array([[0,0],[1,2],[3,2],[-1,0]])

  def test_relative_neighborhood(self):
    pts = np.array([[0,0],[1,2],[3,2],[-1,0]])

if __name__ == '__main__':
  unittest.main()
