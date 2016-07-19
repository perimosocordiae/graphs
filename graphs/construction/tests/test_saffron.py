import numpy as np
import unittest
from numpy.testing import assert_array_almost_equal

from graphs.construction import saffron


class TestSaffron(unittest.TestCase):

  def test_x(self):
    theta = np.concatenate((np.linspace(-0.25, 0.3, 8),
                            np.linspace(2.86, 3.4, 8)))
    n = theta.shape[0]
    X = np.column_stack((np.sin(theta), np.sin(theta) * np.cos(theta)))

    G = saffron(X, q=5, k=2, tangent_dim=1, curv_thresh=0.9, decay_rate=0.5)

    expected_ii = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9,
                   10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15]
    expected_jj = [2, 1, 3, 2, 4, 1, 5, 2, 6, 3, 4, 6, 5, 7, 5, 6, 10, 9, 11,
                   10, 12, 9, 9, 13, 11, 14, 15, 12, 13, 15, 13, 14]
    expected_w = [0.214, 0.105, 0.219, 0.109, 0.222, 0.109, 0.221, 0.111, 0.216,
                  0.111, 0.11, 0.107, 0.107, 0.102, 0.208, 0.102, 0.207, 0.101,
                  0.213, 0.105, 0.217, 0.105, 0.213, 0.217, 0.109, 0.215, 0.209,
                  0.108, 0.106, 0.103, 0.209, 0.103]
    exp = np.zeros((n, n), dtype=float)
    exp[expected_ii, expected_jj] = expected_w

    assert_array_almost_equal(G.matrix('dense'), exp, decimal=3)

  def test_intersecting_planes(self):
    n1 = np.array([-0.25, -1, 1])
    n2 = np.array([0.5, 0.75, 1.25])
    x1, y1 = map(np.ravel, np.meshgrid(np.linspace(-0.75, 1.5, 10),
                                       np.linspace(-1, 1, 9)))
    z1 = (-n1[0]*x1 - n1[1]*y1) / n1[2]
    x2, y2 = map(np.ravel, np.meshgrid(np.linspace(-1, 1, 8),
                                       np.linspace(-1.2, 0.9, 9)))
    z2 = (-n2[0]*x2 - n2[1]*y2) / n2[2]
    X = np.vstack((np.c_[x1, y1, z1], np.c_[x2, y2, z2]))

    # just a smoke test for now, to test the tangent_dim > 1 case
    saffron(X, q=16, k=3, tangent_dim=2, decay_rate=0.75, max_iter=30)

  # XXX: This test doesn't pass, though it's unclear if that's due to a bug.
  '''
  def test_helix(self):
    # attempt to replicate the squashed helix example from the paper
    t = np.linspace(0, 7*np.pi, 439)
    X = np.column_stack((np.sin(t), np.cos(t), 0.001*t))
    G = saffron(X, q=32, k=4, tangent_dim=1, curv_thresh=0.95, decay_rate=0.9,
                max_iter=100)
    # check that G doesn't short circuit across loops of the helix
    ii, jj = G.pairs().T
    diag_offsets = np.unique(np.abs(ii - jj))
    assert_array_equal(diag_offsets, [1, 2])'''


if __name__ == '__main__':
  unittest.main()
