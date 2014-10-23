import matplotlib
matplotlib.use('template')

import numpy as np
import unittest
from numpy.testing import assert_array_equal

from graphs.construction import perturbed_mst, disjoint_mst


class TestSpanningTree(unittest.TestCase):

  def test_perturbed_mst(self):
    pts = np.array([[0,0],[1,2],[3,2],[-1,0]])

  def test_disjoint_mst(self):
    pts = np.array([[0,0],[1,2],[3,2],[-1,0]])


if __name__ == '__main__':
  unittest.main()
