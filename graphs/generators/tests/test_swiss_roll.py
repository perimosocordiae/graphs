import matplotlib
matplotlib.use('template')

import unittest

from graphs.generators import swiss_roll as sr


class TestSwissRoll(unittest.TestCase):

  def test_swiss_roll(self):
    X = sr.swiss_roll(6, 10)
    self.assertEqual(X.shape, (10, 3))
    X, theta = sr.swiss_roll(3.0, 25, theta_noise=0, radius_noise=0,
                             return_theta=True)
    self.assertEqual(X.shape, (25, 3))
    self.assertEqual(theta.shape, (25,))
    self.assertAlmostEqual(theta.max(), 3.0)

  def test_error_ratio(self):
    # TODO: test
    sr.error_ratio


if __name__ == '__main__':
  unittest.main()
