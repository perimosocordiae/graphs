import numpy as np
import unittest
from numpy.testing import assert_array_equal

from graphs.construction import random_graph


class TestRandomGraph(unittest.TestCase):
  def test_random_graph(self):
    for degree in (np.ones(5), [1,2,2], [1,0,0,1]):
      G = random_graph(degree)
      assert_array_equal(degree, G.degree(kind='out'))
      self.assertEqual(1, G.edge_weights().max())

    # Check that degrees >= n will throw an error
    self.assertRaises(ValueError, random_graph, [1,2,3])


if __name__ == '__main__':
  unittest.main()
