import numpy as np
from scipy.spatial import Delaunay
from scipy.sparse import coo_matrix
from graphs import Graph, plot_graph

from common.distance import SquaredL2

__all__ = ['delaunay_graph', 'gabriel_graph', 'relative_neighborhood_graph']


def delaunay_graph(X):
  tri = Delaunay(X)
  n = X.shape[0]
  rows = np.empty(2*np.product(tri.simplices.shape), dtype=np.intp)
  cols = np.empty_like(rows)
  data = np.ones_like(rows, dtype=bool)
  i = 0
  d = tri.simplices.shape[1]
  for corners in tri.simplices:
    rows[i:i+d] = corners
    cols[i:i+d-1] = corners[1:]
    i += d
    cols[i-1] = corners[0]
  rows[i:] = cols[:i]
  cols[i:] = rows[:i]
  adj = coo_matrix((data, (rows, cols)), shape=(n,n))
  return Graph.from_adj_matrix(adj)


def gabriel_graph(X, metric=SquaredL2):
  a,b = np.triu_indices(X.shape[0], k=1)
  midpoints = (X[a] - X[b]) / 2
  Dmid = metric.between(midpoints, X)
  Dedge = metric.pairwise(X[a], X[b])
  mask = np.all(Dedge[:,None] <= Dmid, axis=1)
  pairs = np.transpose((a[mask],b[mask]))
  return Graph.from_edge_pairs(np.vstack((pairs,pairs[:,::-1])))


def relative_neighborhood_graph(X, metric=SquaredL2):
  a,b = np.triu_indices(X.shape[0], k=1)
  Da = metric.between(X[a], X)
  Db = metric.between(X[b], X)
  Dedge = metric.pairwise(X[a], X[b])
  mask = np.all((Dedge[:,None] <= Da) | (Dedge[:,None] <= Db), axis=1)
  pairs = np.transpose((a[mask],b[mask]))
  return Graph.from_edge_pairs(np.vstack((pairs,pairs[:,::-1])))


if __name__ == '__main__':
  from matplotlib.pyplot import subplots
  X = np.random.random((5,2))
  _, axes = subplots(ncols=3)
  plot_graph(X, delaunay_graph(X), title='Delaunay', ax=axes[0])
  plot_graph(X, gabriel_graph(X), title='Gabriel', ax=axes[1])
  plot_graph(X, relative_neighborhood_graph(X),
             title='Relative Neighborhood', ax=axes[2])()
