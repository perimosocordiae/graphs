from __future__ import absolute_import

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.metrics.pairwise import pairwise_distances
from graphs import Graph
from ..mini_six import range

__all__ = ['mst', 'perturbed_mst', 'disjoint_mst']


def mst(X, metric='euclidean'):
  D = _pdist(X, metric)
  mst = minimum_spanning_tree(D, overwrite=(metric!='precomputed'))
  return Graph.from_adj_matrix(mst + mst.T)


def perturbed_mst(X, num_perturbations=20, metric='euclidean', jitter=None):
  '''Builds a graph as the union of several MSTs on perturbed data.
  Reference: http://ecovision.mit.edu/~sloop/shao.pdf, page 8
  jitter refers to the scale of the gaussian noise added for each perturbation.
  When jitter is None, it defaults to the 5th percentile interpoint distance.
  Note that metric cannot be 'precomputed', as multiple MSTs are computed.'''
  D = pairwise_distances(X, metric=metric)
  if jitter is None:
    jitter = np.percentile(D[D>0], 5)
  W = minimum_spanning_tree(D)
  W = W + W.T
  W.data[:] = 1.0  # binarize
  for i in range(num_perturbations):
    pX = X + np.random.normal(scale=jitter, size=X.shape)
    pW = minimum_spanning_tree(pairwise_distances(pX, metric=metric))
    pW = pW + pW.T
    pW.data[:] = 1.0
    W = W + pW
  # final graph is the average over all pertubed MSTs + the original
  W.data /= (num_perturbations + 1.0)
  return Graph.from_adj_matrix(W)


def disjoint_mst(X, num_spanning_trees=3, metric='euclidean'):
  '''Builds a graph as the union of several spanning trees,
  each time removing any edges present in previously-built trees.
  Reference: http://ecovision.mit.edu/~sloop/shao.pdf, page 9.'''
  D = _pdist(X, metric)
  mst = minimum_spanning_tree(D)
  W = mst.copy()
  for i in range(1, num_spanning_trees):
    ii,jj = mst.nonzero()
    D[ii,jj] = np.inf
    D[jj,ii] = np.inf
    mst = minimum_spanning_tree(D)
    W = W + mst
  # MSTs are all one-sided, so we symmetrize here
  return Graph.from_adj_matrix(W + W.T)


def _pdist(X, metric):
  if metric == 'precomputed':
    return X
  return pairwise_distances(X, metric=metric)
