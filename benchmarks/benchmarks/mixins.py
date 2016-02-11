import matplotlib
matplotlib.use('template')
import scipy.sparse as ss
import numpy as np
from graphs import Graph


class Labeling(object):
    params = ['dense', 'coo', 'csr']
    param_names = ['adj_format']

    def setup(self, adj_format):
        n = 500
        density = 0.05
        adj = ss.rand(n, n, density=density)
        if adj_format == 'dense':
            adj = adj.A
        else:
            adj = adj.asformat(adj_format)
        self.G = Graph.from_adj_matrix(adj)
        self.G.symmetrize()
        self.regression_y = np.random.random((n//2,1))
        self.regression_mask = slice(None, None, 2)

    def time_greedy_coloring(self, adj_format):
        self.G.greedy_coloring()

    def time_spectral_clustering(self, adj_format):
        self.G.spectral_clustering(2)

    def time_regression_no_smooth(self, adj_format):
        self.G.regression(self.regression_y, self.regression_mask)

    def time_regression_smooth(self, adj_format):
        self.G.regression(self.regression_y, self.regression_mask,
                          smoothness_penalty=1e-3)
