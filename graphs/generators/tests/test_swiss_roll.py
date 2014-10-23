import matplotlib
matplotlib.use('template')

import numpy as np
import unittest
from numpy.testing import assert_array_equal

from graphs.generators import swiss_roll as sr


class TestSwissRoll(unittest.TestCase):

  def test_swiss_roll(self):
    X = sr.swiss_roll(6, 10)

  def test_error_ratio(self):
    sr.error_ratio


if __name__ == '__main__':
  unittest.main()
