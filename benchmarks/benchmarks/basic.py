import matplotlib
matplotlib.use('template')
import scipy.sparse as ss
from graphs import Graph


class BasicOperations(object):
    params = ['dense', 'coo', 'csr']
    param_names = ['adj_format']

    def setup(self, adj_format):
        n = 1500
        density = 0.2
        adj = ss.rand(n, n, density=density)
        if adj_format == 'dense':
            self.adj = adj.A
        else:
            self.adj = adj.asformat(adj_format)
        self.G = Graph.from_adj_matrix(self.adj)

    def time_construction(self, adj_format):
        Graph.from_adj_matrix(self.adj)

    def time_num_edges(self, adj_format):
        self.G.num_edges()

    def time_num_vertices(self, adj_format):
        self.G.num_vertices()
