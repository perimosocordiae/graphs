import matplotlib
matplotlib.use('template')

import numpy as np
import unittest
from numpy.testing import assert_array_equal

from graphs.generators import mountain_car as mcar


class TestMountainCar(unittest.TestCase):

  def test_traj_sampling(self):
    mcar.sample_mcar_trajectories


if __name__ == '__main__':
  unittest.main()
