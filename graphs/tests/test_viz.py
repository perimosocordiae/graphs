import matplotlib
matplotlib.use('template')

import unittest
import numpy as np
from scipy.sparse import csr_matrix

from graphs.base import (
    EdgePairGraph, DenseAdjacencyMatrixGraph, SparseAdjacencyMatrixGraph)
from graphs.viz import plot_graph


class TestPlot(unittest.TestCase):
  def setUp(self):
    pairs = np.array([[0,1],[0,2],[1,2],[3,4]])
    adj = [[0,1,2,0,0],
           [0,0,3,0,0],
           [0,0,0,0,0],
           [0,0,0,0,4],
           [0,0,0,0,0]]
    self.graphs = [
        EdgePairGraph(pairs),
        DenseAdjacencyMatrixGraph(adj),
        SparseAdjacencyMatrixGraph(csr_matrix(adj))
    ]
    self.coords = np.random.random((5, 3))

  def test_plot_graph_default(self):
    for G in self.graphs:
      plot_graph(G, self.coords[:,:2])
      plot_graph(G, self.coords)

  def test_plot_graph_direction(self):
    for G in self.graphs:
      plot_graph(G, self.coords[:,:2], undirected=True)
      plot_graph(G, self.coords[:,:2], undirected=False)
      plot_graph(G, self.coords, undirected=True)
      plot_graph(G, self.coords, undirected=False)

  def test_plot_graph_weighting(self):
    for G in self.graphs:
      plot_graph(G, self.coords[:,:2], unweighted=True)
      plot_graph(G, self.coords[:,:2], unweighted=False)
      plot_graph(G, self.coords, unweighted=True)
      plot_graph(G, self.coords, unweighted=False)

  def test_plot_graph_styles(self):
    for G in self.graphs:
      plot_graph(G, self.coords, edge_style='r--')
      plot_graph(G, self.coords,
                 edge_style=dict(colors=range(4), linestyles=':'))
      plot_graph(G, self.coords, vertex_style='rx')
      plot_graph(G, self.coords,
                 vertex_style=dict(c=[(0,0,0),(1,1,1)], marker='o'))
      plot_graph(G, self.coords, edge_style='k')
      plot_graph(G, self.coords, edge_style='1')
      with self.assertRaises(ValueError):
        plot_graph(G, self.coords, edge_style='5')
      plot_graph(G, self.coords, edge_style=' x')
      plot_graph(G, self.coords, edge_style='-.')
      plot_graph(G, self.coords, edge_style='k-')


if __name__ == '__main__':
  unittest.main()
