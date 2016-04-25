import numpy as np
from sklearn.metrics import pairwise_distances
import graphs.construction as gc


class KNeighbors(object):
    n = 500
    params = [[1, 10, 100], ['none', 'binary']]
    param_names = ['k', 'weighting']

    def setup(self, k, weighting):
        self.X = np.random.random((self.n, 3))
        self.D = pairwise_distances(self.X)

    def time_knn(self, k, weighting):
        gc.neighbor_graph(self.X, k=k, weighting=weighting)

    def time_knn_precomputed(self, k, weighting):
        gc.neighbor_graph(self.D, k=k, weighting=weighting, precomputed=True)
