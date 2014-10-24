import matplotlib
matplotlib.use('template')

import unittest
from numpy.testing import assert_array_equal

from graphs.generators import shapes, trajectories as traj


class TestTrajectories(unittest.TestCase):

  def test_concat_trajectories(self):
    expected = [[0,1,0,0,0],[1,0,0,0,0],[0,0,0,1,0],[0,0,1,0,1],[0,0,0,1,0]]
    G = traj.concat_trajectories([2, 3], directed=False)
    assert_array_equal(G.matrix(dense=True), expected)

  def test_chunk_up(self):
    T = shapes.SCurve().trajectories(2, 4)
    expected = [[0,1,0,0,0,0,0,0],
                [1,0,1,0,0,0,0,0],
                [0,1,0,1,0,0,0,0],
                [0,0,1,0,0,0,0,0],
                [0,0,0,0,0,1,0,0],
                [0,0,0,0,1,0,1,0],
                [0,0,0,0,0,1,0,1],
                [0,0,0,0,0,0,1,0]]
    G = traj.chunk_up(T, directed=False)
    assert_array_equal(G.matrix(dense=True), expected)
    expected = [[0,1,0,0,0,0,0,0],
                [1,0,0,0,0,0,0,0],
                [0,0,0,1,0,0,0,0],
                [0,0,1,0,0,0,0,0],
                [0,0,0,0,0,1,0,0],
                [0,0,0,0,1,0,0,0],
                [0,0,0,0,0,0,0,1],
                [0,0,0,0,0,0,1,0]]
    G = traj.chunk_up(T, chunk_size=2, directed=False)
    assert_array_equal(G.matrix(dense=True), expected)

if __name__ == '__main__':
  unittest.main()
