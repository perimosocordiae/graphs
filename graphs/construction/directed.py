import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from graphs import Graph, plot_graph
from neighbors import neighbor_graph


def jeff_graph(trajectories, k=5, verbose=False, pruning_thresh=0):
  '''Directed graph construction alg. from Johns & Mahadevan, ICML '07.
  trajectories: list of NxD arrays of ordered states
  '''
  X = np.vstack(trajectories)
  W = neighbor_graph(X, k=k, symmetrize=False)
  if pruning_thresh > 0:
    jeff_prune_edges(X, W, map(len, trajectories), pruning_thresh,
                     copy=False, verbose=verbose)
  return W, X


def jeff_prune_edges(X, G, traj_lengths, pruning_thresh=0.1, verbose=False):
  '''Prune edges in graph G via cosine distance with trajectory edges.'''
  W = G.matrix(dense=True).copy()
  i = 0
  num_bad = 0
  for n in traj_lengths:
    s,t = np.nonzero(W[i:i+n])
    s = s + i
    v = X[t] - X[s]
    v_a = np.diff(X[i:i+n], axis=0)
    theta = pairwise_distances(v, v_a, 'cosine')
    bad_edges = theta > pruning_thresh
    W[s[bad_edges],t[bad_edges]] = 0
    if verbose:
      num_bad += np.count_nonzero(bad_edges)
    i += n
  if verbose:
    print 'removed %d bad edges' % num_bad
  return Graph.from_adj_matrix(W)


def demo_mcar(num_traj=100):
  from matplotlib import pyplot
  from downsample import downsample_trajectories
  from graphs.generators.mountain_car import sample_mcar_trajectories

  trajectories = sample_mcar_trajectories(num_traj, min_length=1)
  print sum(map(len, trajectories)), 'total samples'

  # downsample
  trajectories = downsample_trajectories(trajectories, 0.0001)
  print sum(map(len, trajectories)), 'after downsampling'

  # build graphs
  W_knn, X = jeff_graph(trajectories, k=5, verbose=True)
  W_jeff = jeff_prune_edges(X, W_knn, map(len, trajectories),
                            pruning_thresh=0.1, verbose=True)

  # compare
  _, (ax1, ax2) = pyplot.subplots(ncols=2)
  plot_graph(W_knn, X, title='knn', ax=ax1)
  plot_graph(W_jeff, X, title='Jeff', ax=ax2)()

if __name__ == '__main__':
  demo_mcar()
