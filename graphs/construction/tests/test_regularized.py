import numpy as np
np.set_printoptions(suppress=True, precision=3)
import unittest
import warnings
from numpy.testing import assert_array_almost_equal
from sklearn.utils import ConvergenceWarning

from graphs.construction import sparse_regularized_graph


class TestRegularized(unittest.TestCase):
  def setUp(self):
    np.random.seed(1234)
    self.pts = _gaussian_clusters(2, 5, 20)

  def test_L1_graph(self):
    expected = [
        [0,    0.286,0.352,0.362,0,    0,    0,    0,    0,    0],
        [0.637,0,    0.209,0,    0.153,0,    0,    0,    0,    0],
        [0.446,0.133,0,    0,    0.421,0,    0,    0,    0,    0],
        [0.493,0,    0,    0,    0.507,0,    0,    0,    0,    0],
        [0,    0.062,0.493,0.444,0,    0,    0,    0,    0,    0],
        [0,    0,    0,    0,    0,    0,    0.906,0.055,0.039,0],
        [0,    0,    0,    0,    0,    0.603,0,    0,    0.136,0.261],
        [0,    0,    0,    0,    0,    0.172,0,    0,    0.332,0.496],
        [0,    0,    0,    0,    0,    0.007,0.576,0.278,0,    0.139],
        [0,    0,    0,    0,    0,    0,    0.441,0.395,0.164,0]
    ]
    G = sparse_regularized_graph(self.pts, alpha=0.005)
    assert_array_almost_equal(G.matrix(dense=True), expected, decimal=3)

  def test_L1_knn_graph(self):
    expected = [
        [0,    0.286,0.352,0.362,0,    0,    0,    0,    0,    0],
        [0.637,0,    0.209,0,    0.153,0,    0,    0,    0,    0],
        [0.446,0.133,0,    0,    0.421,0,    0,    0,    0,    0],
        [0.493,0,    0,    0,    0.507,0,    0,    0,    0,    0],
        [0,    0,    0.535,0.465,0,    0,    0,    0,    0,    0],
        [0,    0,    0,    0,    0,    0,    0.924,0,    0.076,0],
        [0,    0,    0,    0,    0,    0.603,0,    0,    0.136,0.261],
        [0,    0,    0,    0,    0,    0,    0,    0,    0.454,0.546],
        [0,    0,    0,    0,    0,    0.138,0.520,0,    0,    0.343],
        [0,    0,    0,    0,    0,    0,    0.441,0.395,0.164,0]
    ]
    G = sparse_regularized_graph(self.pts, alpha=0.005, k=3)
    assert_array_almost_equal(G.matrix(dense=True), expected, decimal=3)

  def test_L1_graph_cv(self):
    expected = [
        [0,    0.231,0.372,0.397,0,    0,    0,    0,    0,    0],
        [0.670,0,    0.205,0,    0.124,0,    0,    0,    0,    0],
        [0.437,0.138,0,    0.012,0.413,0,    0,    0,    0,    0],
        [0.503,0,    0,    0,    0.497,0,    0,    0,    0,    0],
        [0,    0.053,0.509,0.438,0,    0,    0,    0,    0,    0],
        [0,    0,    0,    0,    0,    0,    0.914,0.061,0.025,0],
        [0,    0,    0,    0,    0,    0.597,0,    0,    0.139,0.264],
        [0,    0,    0,    0,    0,    0.311,0,    0,    0.391,0.297],
        [0,    0,    0,    0,    0,    0.043,0.544,0.310,0,    0.103],
        [0,    0,    0,    0,    0,    0,    0.428,0.399,0.173,0]
    ]
    with warnings.catch_warnings():
      warnings.filterwarnings('ignore', category=ConvergenceWarning)
      G = sparse_regularized_graph(self.pts)
    assert_array_almost_equal(G.matrix(dense=True), expected, decimal=3)

  def test_SPD_graph(self):
    expected = [
        [0,    0.216,0.380,0.404,0,    0,    0,    0,    0,    0],
        [0.676,0,    0.123,0,    0.202,0,    0,    0,    0,    0],
        [0.377,0.140,0,    0,    0.483,0,    0,    0,    0,    0],
        [0.506,0,    0,    0,    0.441,0,    0,    0,    0.053,0],
        [0.017,0.065,0.454,0.464,0,    0,    0,    0,    0,    0],
        [0,    0,    0,    0,    0,    0,    0.907,0.093,0,    0],
        [0,    0,    0,    0,    0,    0.575,0,    0,    0.117,0.308],
        [0,    0,    0,    0,    0,    0.295,0,    0,    0.319,0.386],
        [0,    0,    0,    0,    0,    0.010,0.599,0.274,0,    0.117],
        [0,    0,    0,    0,    0,    0,    0.440,0.386,0.174,0]
    ]
    G = sparse_regularized_graph(self.pts, positive=True, alpha=0.002)
    assert_array_almost_equal(G.matrix(dense=True), expected, decimal=3)

  def test_SPD_graph_cv(self):
    expected = [
        [0,    0.230,0.380,0.390,0,    0,    0,    0,    0,    0],
        [0.603,0,    0.209,0,    0.188,0,    0,    0,    0,    0],
        [0.366,0.133,0,    0,    0.501,0,    0,    0,    0,    0],
        [0.414,0,    0.119,0,    0.383,0,    0,    0,    0.084,0],
        [0.002,0.062,0.482,0.454,0,    0,    0,    0,    0,    0],
        [0,    0,    0,    0,    0,    0,    0.921,0.079,0,    0],
        [0,    0,    0,    0,    0.006,0.584,0,    0,    0.088,0.322],
        [0,    0,    0,    0,    0,    0.286,0,    0,    0.288,0.426],
        [0,    0,    0,    0,    0,    0.052,0.541,0.254,0,    0.153],
        [0,    0,    0,    0,    0,    0,    0.458,0.408,0.134,0]
    ]
    G = sparse_regularized_graph(self.pts, positive=True)
    assert_array_almost_equal(G.matrix(dense=True), expected, decimal=3)


def _gaussian_clusters(num_clusters, pts_per_cluster, dim):
  n = num_clusters * pts_per_cluster
  offsets = np.random.uniform(-9, 9, (num_clusters, dim))
  return np.random.randn(n, dim) + np.repeat(offsets, pts_per_cluster, axis=0)


if __name__ == '__main__':
  unittest.main()
