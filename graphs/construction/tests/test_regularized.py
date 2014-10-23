import matplotlib
matplotlib.use('template')

import numpy as np
import unittest
from numpy.testing import assert_array_equal

from graphs.construction import sparse_regularized_graph


class TestRegularized(unittest.TestCase):

  def test_sparse_regularized_graph(self):
    pts = np.array([[0,0],[1,2],[3,2],[-1,0]])


if __name__ == '__main__':
  unittest.main()
