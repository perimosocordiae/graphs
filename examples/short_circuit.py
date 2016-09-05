from __future__ import print_function
import numpy as np
from matplotlib import pyplot as plt

from graphs.datasets import swiss_roll
from graphs.construction import neighbor_graph


def main():
  np.random.seed(1234)
  X, theta = swiss_roll(8, 300, return_theta=True, radius=0.5)
  GT = np.column_stack((theta, X[:,1]))
  g = neighbor_graph(X, k=6)
  g = g.from_adj_matrix(g.matrix('dense'))
  ct = 12

  _, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8),
                         sharex=True, sharey=True)
  _plot_diff(axes[0,0], GT, g, g.minimum_spanning_subtree(), title='MST')
  _plot_diff(axes[0,1], GT, g, g.circle_tear(cycle_len_thresh=ct),
             title='Circle Tear (%d)' % ct)
  _plot_diff(axes[1,0], GT, g, g.cycle_cut(cycle_len_thresh=ct),
             title='Cycle Cut (%d)' % ct)
  _plot_diff(axes[1,1], GT, g, g.isograph(), title='Isograph')
  plt.show()


def _plot_diff(ax, x, g1, g2, title=''):
  g1.plot(x, ax=ax, edge_style='y-', vertex_style='k.')
  g2.plot(x, ax=ax, edge_style='b-', vertex_style='k.')
  ax.set_title(title)


if __name__ == '__main__':
  main()
