from __future__ import absolute_import
import numpy as np
import unittest
from numpy.testing import assert_array_equal

from .. import chain_graph, lattice_graph


class TestStructured(unittest.TestCase):

  def test_chain_graph(self):
    expected = np.diag([1,1], k=1)
    g = chain_graph(3, directed=True)
    assert_array_equal(g.matrix('dense'), expected)

    expected += np.diag([1,1], k=-1)
    g = chain_graph(3, wraparound=False)
    assert_array_equal(g.matrix('dense'), expected)

    expected[0,2] = 1
    expected[2,0] = 1
    g = chain_graph(3, wraparound=True)
    assert_array_equal(g.matrix('dense'), expected)

  def test_lattice_graph(self):
    self.assertRaises(ValueError, lattice_graph, [])

    expected = np.diag([1,1], k=1) + np.diag([1,1], k=-1)
    g = lattice_graph((3,), wraparound=False)
    assert_array_equal(g.matrix('dense'), expected)

    expected = np.diag([1,1,0,1,1], k=1) + np.diag([1,1,0,1,1], k=-1)
    expected += np.diag([1,1,1], k=3) + np.diag([1,1,1], k=-3)
    g = lattice_graph((3,2), wraparound=False)
    assert_array_equal(g.matrix('dense'), expected)

    expected[[0,3],[2,5]] = 1
    expected[[2,5],[0,3]] = 1
    g = lattice_graph((3,2), wraparound=True)
    assert_array_equal(g.matrix('dense'), expected)

    expected = np.diag([1,1,1], k=1) + np.diag([1,1,1], k=-1)
    g = lattice_graph((1,4), wraparound=False)
    assert_array_equal(g.matrix('dense'), expected)


if __name__ == '__main__':
  unittest.main()
