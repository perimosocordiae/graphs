import numpy as np
import scipy.linalg as sl
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

  def regression(self, y, y_mask, smoothness_penalty=0, kernel='rbf'):
    '''Perform vertex-valued regression, given partial labels.
    y : (n,d) array of known labels
    y_mask : index object such that all_labels[y_mask] == y

    From "Regularization and Semi-supervised Learning on Large Graphs"
      by Belkin, Matveeva, and Niyogi in 2004.
    Doesn't support multiple labels per vertex, unlike the paper's algorithm.
    To allow provided y values to change, use a (small) smoothness_penalty.
    '''
    n = self.num_vertices()

    # input validation for y
    y = np.array(y, copy=True)
    ravel_f = False
    if y.ndim == 1:
      y = y[:,None]
      ravel_f = True
    if y.ndim != 2 or y.size == 0:
      raise ValueError('Invalid shape of y array: %s' % (y.shape,))
    k, d = y.shape

    # input validation for y_mask
    if not hasattr(y_mask, 'dtype') or y_mask.dtype != 'bool':
      tmp = np.zeros(n, dtype=bool)
      tmp[y_mask] = True
      y_mask = tmp

    # mean-center known y for stability
    y_mean = y.mean(axis=0)
    y -= y_mean

    # use the normalized Laplacian for the smoothness matrix
    S = ssc.laplacian(self._kernel_matrix(kernel), normed=True)
    if ss.issparse(S):
      S = S.tocsr()

    if smoothness_penalty == 0:
      # see Algorithm 2: Interpolated Regularization
      unlabeled_mask = ~y_mask
      S_23 = S[unlabeled_mask, :]
      S_3 = S_23[:, unlabeled_mask]
      rhs = S_23[:, y_mask].dot(y)
      if ss.issparse(S):
        f_unlabeled = ss.linalg.spsolve(S_3, rhs)
        if f_unlabeled.ndim == 1:
          f_unlabeled = f_unlabeled[:,None]
      else:
        f_unlabeled = sl.solve(S_3, rhs, sym_pos=True, overwrite_a=True,
                               overwrite_b=True)
      f = np.zeros((n, d))
      f[y_mask] = y
      f[unlabeled_mask] = -f_unlabeled
    else:
      # see Algorithm 1: Tikhonov Regularization in the paper
      y_hat = np.zeros((n, d))
      y_hat[y_mask] = y
      I = y_mask.astype(float)  # only one label per vertex
      lhs = k * smoothness_penalty * S
      if ss.issparse(lhs):
        lhs.setdiag(lhs.diagonal() + I)
        f = ss.linalg.lsqr(lhs, y_hat)[0]
      else:
        lhs.flat[::n+1] += I
        f = sl.solve(lhs, y_hat, sym_pos=True, overwrite_a=True,
                     overwrite_b=True)

    # re-add the mean
    f += y_mean
    if ravel_f:
      return f.ravel()
    return f

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
