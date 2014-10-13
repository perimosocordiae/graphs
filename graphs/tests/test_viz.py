import unittest
import numpy as np

from graphs.base import EdgePairGraph, DenseAdjacencyMatrixGraph
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
        DenseAdjacencyMatrixGraph(adj)
    ]
    self.coords = np.random.random((5, 2))

  def test_plot_graph_default(self):
    for G in self.graphs:
      plot_graph(G, self.coords)

  def test_plot_graph_undirected(self):
    for G in self.graphs:
      plot_graph(G, self.coords, undirected=True)

  def test_plot_graph_unweighted(self):
    for G in self.graphs:
      plot_graph(G, self.coords, unweighted=True)

  def test_plot_graph_styles(self):
    for G in self.graphs:
      plot_graph(G, self.coords, edge_style='r--')
      plot_graph(G, self.coords,
                 edge_style=dict(colors=range(4), linestyles=':'))
      plot_graph(G, self.coords, vertex_style='rx')
      plot_graph(G, self.coords,
                 vertex_style=dict(c=[(0,0,0),(1,1,1)], marker='o'))


if __name__ == '__main__':
  unittest.main()
