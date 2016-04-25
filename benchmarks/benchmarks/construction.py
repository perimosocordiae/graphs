import numpy as np
from sklearn.metrics import pairwise_distances
import graphs.construction as gc


class Neighbors(object):
    n = 500
    params = [[None, 0.25, 0.5], [None, 1, 100], ['none', 'binary']]
    param_names = ['epsilon', 'k', 'weighting']

    def setup(self, epsilon, k, weighting):
        if epsilon is None and k is None:
            raise NotImplementedError()
        self.X = np.random.random((self.n, 3))
        self.D = pairwise_distances(self.X)

    def time_neighbor_graph(self, epsilon, k, weighting):
        gc.neighbor_graph(self.X, k=k, epsilon=epsilon, weighting=weighting)

    def time_neighbor_graph_precomputed(self, epsilon, k, weighting):
        gc.neighbor_graph(self.D, k=k, epsilon=epsilon, weighting=weighting,
                          precomputed=True)
