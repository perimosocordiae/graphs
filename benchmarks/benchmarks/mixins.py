import matplotlib
matplotlib.use('template')
import scipy.sparse as ss
import numpy as np
from graphs import Graph


class _RandomFormatsBase(object):
    n = 500
    density = 0.05
    params = ['dense', 'coo', 'csr']
    param_names = ['adj_format']

    def setup(self, adj_format, *args):
        adj = ss.rand(self.n, self.n, density=self.density, random_state=1234)
        if adj_format == 'dense':
            adj = adj.A
        else:
            adj = adj.asformat(adj_format)
        self.G = Graph.from_adj_matrix(adj)
        self.G.symmetrize()


class Labeling(_RandomFormatsBase):
    def time_greedy_coloring(self, *args):
        self.G.greedy_coloring()

    def time_spectral_clustering(self, *args):
        self.G.spectral_clustering(2)


class Regression(_RandomFormatsBase):
    params = [['dense', 'coo', 'csr'], ['rbf', 'none', 'binary'], [0, 1e-3]]
    param_names = ['adj_format', 'kernel', 'smoothness_penalty']

    def setup(self, *args):
        _RandomFormatsBase.setup(self, *args)
        self.regression_y = np.random.random((self.n//2, 1))
        self.regression_mask = slice(None, None, 2)

    def time_regression(self, _, k, s):
        self.G.regression(self.regression_y, self.regression_mask,
                          smoothness_penalty=s, kernel=k)
