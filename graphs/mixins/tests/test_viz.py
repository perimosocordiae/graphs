import unittest
import numpy as np
from io import BytesIO
from scipy.sparse import csr_matrix
from matplotlib import pyplot
pyplot.switch_backend('template')

from graphs import Graph


class TestPlot(unittest.TestCase):
  def setUp(self):
    pairs = np.array([[0,1],[0,2],[1,2],[3,4]])
    adj = [[0,1,2,0,0],
           [0,0,3,0,0],
           [0,0,0,0,0],
           [0,0,0,0,4],
           [0,0,0,0,0]]
    self.graphs = [
        Graph.from_edge_pairs(pairs),
        Graph.from_edge_pairs(pairs, symmetric=True),
        Graph.from_adj_matrix(adj),
        Graph.from_adj_matrix(csr_matrix(adj)),
    ]
    self.coords = np.random.random((5, 3))

  def test_plot_default(self):
    for G in self.graphs:
      G.plot(self.coords[:,:1])  # 1d plotting
      G.plot(self.coords[:,:2])  # 2d plotting
      G.plot(self.coords)        # 3d plotting

  def test_plot_direction(self):
    for G in self.graphs:
      G.plot(self.coords[:,:2], directed=True)
      G.plot(self.coords[:,:2], directed=False)
      G.plot(self.coords, directed=True)
      G.plot(self.coords, directed=False)

  def test_plot_weighting(self):
    for G in self.graphs:
      G.plot(self.coords[:,:2], weighted=True)
      G.plot(self.coords[:,:2], weighted=False)
      G.plot(self.coords, weighted=True)
      G.plot(self.coords, weighted=False)

  def test_plot_styles(self):
    x = self.coords[:,:2]  # use 2d coords, 3d _get_axis is slow
    for G in self.graphs:
      G.plot(x, edge_style='r--')
      G.plot(x, edge_style=dict(colors=[0,1,2,3], linestyles=':'))
      G.plot(x, vertex_style='rx')
      G.plot(x, vertex_style=dict(c=[(0,0,0),(1,1,1)], marker='o'))
      G.plot(x, edge_style='k')
      G.plot(x, edge_style='1')
      G.plot(x, edge_style='01')
      G.plot(x, edge_style=' x')
      G.plot(x, edge_style='-.')
      G.plot(x, edge_style='k-')
      # Make sure we break with bogus styles
      with self.assertRaises(ValueError):
        G.plot(x, edge_style='z')
      with self.assertRaises(ValueError):
        G.plot(x, edge_style='::')
      with self.assertRaises(ValueError):
        G.plot(x, edge_style='oo')
      with self.assertRaises(ValueError):
        G.plot(x, edge_style='kk')

  def test_plot_fig(self):
    for G in self.graphs:
      G.plot(self.coords[:,:2], fig='new')
      G.plot(self.coords[:,:2], fig='current')

  def test_to_html(self):
    for G in self.graphs:
      buf = BytesIO()
      # just make sure no exceptions are thrown
      G.to_html(buf, directed=False)
      buf.truncate(0)

    c = np.arange(5)
    G.to_html(buf, vertex_ids=c, directed=False, title='Test Page')
    buf.truncate(0)
    G.to_html(buf, vertex_colors=c, directed=False)
    buf.truncate(0)
    G.to_html(buf, vertex_labels=c, directed=False)
    buf.truncate(0)
    with self.assertRaises(ValueError):
      G.to_html(buf, vertex_colors=c, vertex_labels=c, directed=False)
    with self.assertRaises(ValueError):
      G.to_html(buf, vertex_ids=c[:2], directed=False)
    with self.assertRaises(ValueError):
      G.to_html(buf, vertex_colors=c[:2], directed=False)
    with self.assertRaises(ValueError):
      G.to_html(buf, vertex_labels=c[:2], directed=False)


if __name__ == '__main__':
  unittest.main()
