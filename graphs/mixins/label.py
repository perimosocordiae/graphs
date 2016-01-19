import numpy as np
import scipy.sparse as ss
import scipy.sparse.csgraph as ssc
import warnings
from itertools import count
from sklearn.cluster import spectral_clustering


class LabelMixin(object):

  def greedy_coloring(self):
    '''Returns a greedy vertex coloring, as an array of ints.'''
    n = self.num_vertices()
    coloring = np.zeros(n, dtype=int)
    for i, nbrs in enumerate(self.adj_list()):
      nbr_colors = set(coloring[nbrs])
      for c in count(1):
        if c not in nbr_colors:
          coloring[i] = c
          break
    return coloring

  def spectral_clustering(self, num_clusters, kernel='rbf'):
    aff = self._kernel_matrix(kernel)
    return spectral_clustering(aff, n_clusters=num_clusters)

  def spread_labels(self, partial_labels, kernel='rbf', alpha=0.2, tol=1e-3,
                    max_iter=30):
    # compute the gram matrix
    gram = -ssc.laplacian(self._kernel_matrix(kernel), normed=True)
    if ss.issparse(gram):
      gram.data[gram.row == gram.col] = 0
    else:
      np.fill_diagonal(gram, 0)

    # initialize label distributions
    partial_labels = np.asarray(partial_labels)
    unlabeled = partial_labels == -1
    classes = np.unique(partial_labels[~unlabeled])
    label_dists = np.zeros((len(partial_labels), len(classes)), dtype=int)
    for label in classes:
      label_dists[partial_labels==label, classes==label] = 1

    # initialize clamping terms
    clamp_weights = np.where(unlabeled, alpha, 1)
    y_static = label_dists * min(1 - alpha, 1)
    y_static[unlabeled] = 0

    # iterate
    for it in range(max_iter):
      old_label_dists = label_dists
      label_dists = gram.dot(label_dists)
      label_dists *= clamp_weights[:,None]
      label_dists += y_static
      # check convergence
      if np.abs(label_dists - old_label_dists).sum() <= tol:
        break
    else:
      warnings.warn("spread_labels didn't converge in %d iterations" % max_iter)

    return classes[label_dists.argmax(axis=1)]

  def _kernel_matrix(self, kernel):
    if kernel == 'none':
      return self.matrix(csr=True, dense=True)
    # make a copy to modify with the kernel
    aff = self.matrix(csr=True, copy=True)
    if kernel == 'rbf':
      aff.data = np.exp(-aff.data / aff.data.std())
    elif kernel == 'binary':
      aff.data[:] = 1
    else:
      raise ValueError('Invalid kernel type: %r' % kernel)
    return aff
