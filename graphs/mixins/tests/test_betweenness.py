import unittest
import numpy as np
from numpy.testing import assert_array_equal

# Test the non-Cython version specifically
from graphs.mixins._betweenness import _betweenness

ADJ = np.array([[0,1,2,0],
                [1,0,0,3],
                [2,0,0,1],
                [0,3,1,0]])


class TestBetweenness(unittest.TestCase):

  def test_betweenness_edge_unweighted(self):
    res = _betweenness(ADJ, False, False)
    assert_array_equal(res, [2,2,2,2,2,2,2,2])

  def test_betweenness_edge_weighted(self):
    res = _betweenness(ADJ, True, False)
    assert_array_equal(res, [2,3,2,1,3,2,1,2])

  def test_betweenness_vertex_unweighted(self):
    res = _betweenness(ADJ, False, True)
    assert_array_equal(res, [1,1,1,1])

  def test_betweenness_vertex_weighted(self):
    res = _betweenness(ADJ, True, True)
    assert_array_equal(res, [2,0,2,0])


if __name__ == '__main__':
  unittest.main()
