from __future__ import print_function
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import pairwise_distances
from time import time

from graphs.datasets.swiss_roll import swiss_roll
from graphs.construction import (
    neighbor_graph, b_matching, gabriel_graph,
    relative_neighborhood_graph, manifold_spanning_graph,
    sparse_regularized_graph, saffron, mst, disjoint_mst
)


def main():
  X, theta = swiss_roll(8, 500, return_theta=True)
  D = pairwise_distances(X)
  graph_info = [
    _c('5-NN', neighbor_graph, D, k=6, precomputed=True),
    _c('b-matching', b_matching, D, 6),
    _c('gabriel', gabriel_graph, X),
    _c('rel. neighborhood', relative_neighborhood_graph,D,metric='precomputed'),
    _c('manifold spanning', manifold_spanning_graph, X, 2),
    _c('L1', sparse_regularized_graph, X, kmax=10, sparsity_param=0.0005),
    _c('SAFFRON', saffron, X, q=15, k=5, tangent_dim=2),
    _c('MST', mst, D, metric='precomputed'),
    _c('dMST', disjoint_mst, D, metric='precomputed'),
  ]

  print('Plotting graphs & embeddings')
  fig1, axes1 = plt.subplots(nrows=3, ncols=3, subplot_kw=dict(projection='3d'))
  fig2, axes2 = plt.subplots(nrows=3, ncols=3)
  fig1.suptitle('Original Coordinates')
  fig2.suptitle('Isomap Embeddings')

  for ax1, ax2, info in zip(axes1.flat, axes2.flat, graph_info):
    label, G, gg, emb, mask = info
    G.plot(X, ax=ax1, title=label, vertex_style=dict(c=theta))
    gg.plot(emb, ax=ax2, title=label, vertex_style=dict(c=theta[mask]))
    ax1.view_init(elev=5, azim=70)
    ax1.set_axis_off()
    ax2.set_axis_off()
  plt.show()


def _c(label, fn, *args, **kwargs):
  print('Constructing', label, 'graph:')
  tic = time()
  G = fn(*args, **kwargs)
  print('  -> took %.3f secs' % (time() - tic))
  num_ccs, labels = G.connected_components(directed=False)
  if num_ccs == 1:
    mask = Ellipsis
    gg = G
  else:
    mask = labels == np.bincount(labels).argmax()
    gg = G.subgraph(mask)
  emb = gg.isomap(num_dims=2, directed=False)
  return label, G, gg, emb, mask


if __name__ == '__main__':
  main()
