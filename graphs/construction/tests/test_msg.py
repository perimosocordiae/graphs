import numpy as np
import unittest
from numpy.testing import assert_array_equal

from graphs.generators.swiss_roll import swiss_roll, error_ratio
from graphs.construction import manifold_spanning_graph


class TestMSG(unittest.TestCase):

  def test_swiss_roll(self):
    np.random.seed(1234)
    X, theta = swiss_roll(6, 120, radius=4.8, return_theta=True)
    GT = np.hstack((theta[:,None], X[:,1:2]))
    GT -= GT.min(axis=0)
    GT /= GT.max(axis=0)

    G = manifold_spanning_graph(X, 2)
    self.assertEqual(error_ratio(G, GT), 0.0)

  def test_two_moons(self):
    np.random.seed(1234)
    n1,n2 = 55,75
    theta = np.hstack((np.random.uniform(0, 1, size=n1),
                       np.random.uniform(1, 2, size=n2))) * np.pi
    r = 1.3 + 0.12 * np.random.randn(n1+n2)[:,None]
    X = r * np.hstack((np.cos(theta), np.sin(theta))).reshape((-1,2), order='F')
    X[:n1] += np.array([[0, -0.2]])
    X[n1:] += np.array([[0.9, 0.25]])

    G = manifold_spanning_graph(X, 2, num_ccs=2)
    num_ccs, labels = G.connected_components()
    self.assertEqual(num_ccs, 2)
    assert_array_equal(labels[:n1], np.zeros(n1))
    assert_array_equal(labels[n1:], np.ones(n2))

if __name__ == '__main__':
  unittest.main()
