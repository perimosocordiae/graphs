import unittest

from graphs.generators import shapes


class TestShapes(unittest.TestCase):

  def test_mobius(self):
    S = shapes.MobiusStrip(radius=1.0, max_width=1.0)
    X = S.point_cloud(25)
    T = S.trajectories(2, 10)
    self.assertEqual(X.shape, (25, 3))
    self.assertEqual(len(T), 2)
    self.assertEqual(T[0].shape, (10, 3))
    self.assertEqual(T[1].shape, (10, 3))

  def test_s_curve(self):
    S = shapes.SCurve(radius=1.0)
    X = S.point_cloud(25)
    T = S.trajectories(2, 10)
    self.assertEqual(X.shape, (25, 3))
    self.assertEqual(len(T), 2)
    self.assertEqual(T[0].shape, (10, 3))
    self.assertEqual(T[1].shape, (10, 3))

  def test_figure_eight(self):
    for d in (2,3):
      S = shapes.FigureEight(radius=1.0, dimension=d)
      X = S.point_cloud(25)
      T = S.trajectories(2, 10)
      self.assertEqual(X.shape, (25, d))
      self.assertEqual(len(T), 2)
      self.assertEqual(T[0].shape, (10, d))
      self.assertEqual(T[1].shape, (10, d))

if __name__ == '__main__':
  unittest.main()
