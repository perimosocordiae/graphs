import matplotlib
matplotlib.use('template')

import numpy as np
import unittest
from numpy.testing import assert_array_equal

from graphs.generators import trajectories as traj


class TestTrajectories(unittest.TestCase):

  def test_concat_trajectories(self):
    traj.concat_trajectories

  def test_chunk_up(self):
    traj.chunk_up


if __name__ == '__main__':
  unittest.main()
