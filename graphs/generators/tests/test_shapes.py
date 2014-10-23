import matplotlib
matplotlib.use('template')

import numpy as np
import unittest
from numpy.testing import assert_array_equal

from graphs.generators import shapes


class TestShapes(unittest.TestCase):

  def test_mobius(self):
    shapes.MobiusStrip

  def test_s_curve(self):
    shapes.SCurve

  def test_figure_eight(self):
    shapes.FigureEight

if __name__ == '__main__':
  unittest.main()
