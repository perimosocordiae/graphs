from __future__ import absolute_import
import numpy as np
import scipy.linalg as sl
import scipy.sparse as ss
import warnings
from itertools import count
from sklearn.cluster import spectral_clustering

from ..mini_six import range


class LabelMixin(object):

  def color_greedy(self):
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

  def cluster_spectral(self, num_clusters, kernel='rbf'):
    aff = self.kernelize(kernel).matrix()
    return spectral_clustering(aff, n_clusters=num_clusters)

  def classify_nearest(self, partial_labels):
    '''Simple semi-supervised classification, by assigning unlabeled vertices
    the label of nearest labeled vertex.

    partial_labels: (n,) array of integer labels, -1 for unlabeled.
    '''
    labels = np.array(partial_labels, copy=True)
    unlabeled = labels == -1
    # compute geodesic distances from unlabeled vertices
    D_unlabeled = self.shortest_path()[unlabeled]
    # set distances to other unlabeled vertices to infinity
    D_unlabeled[:,unlabeled] = np.inf
    # find shortest distances to labeled vertices
    idx = D_unlabeled.argmin(axis=1)
    # apply the label of the closest vertex
    labels[unlabeled] = labels[idx]
    return labels

  def classify_lgc(self, partial_labels, kernel='rbf', alpha=0.2, tol=1e-3,
                   max_iter=30):
    '''Iterative label spreading for semi-supervised classification.

    partial_labels: (n,) array of integer labels, -1 for unlabeled.
    kernel: one of {'none', 'rbf', 'binary'}, for reweighting edges.
    alpha: scalar, clamping factor.
    tol: scalar, convergence tolerance.
    max_iter: integer, cap on the number of iterations performed.

    From "Learning with local and global consistency"
      by Zhou et al. in 2004.

    Based on the LabelSpreading implementation in scikit-learn.
    '''
    # compute the gram matrix
    gram = -self.kernelize(kernel).laplacian(normed=True)
    if ss.issparse(gram):
      gram.data[gram.row == gram.col] = 0
    else:
      np.fill_diagonal(gram, 0)

    # initialize label distributions
    partial_labels = np.asarray(partial_labels)
    unlabeled = partial_labels == -1
    label_dists, classes = _onehot(partial_labels, mask=~unlabeled)

    # initialize clamping terms
    clamp_weights = np.where(unlabeled, alpha, 1)[:,None]
    y_static = label_dists * min(1 - alpha, 1)

    # iterate
    for it in range(max_iter):
      old_label_dists = label_dists
      label_dists = gram.dot(label_dists)
      label_dists *= clamp_weights
      label_dists += y_static
      # check convergence
      if np.abs(label_dists - old_label_dists).sum() <= tol:
        break
    else:
      warnings.warn("spread_labels didn't converge in %d iterations" % max_iter)

    return classes[label_dists.argmax(axis=1)]

  def classify_local(self, partial_labels, C_l=10.0, C_u=1e-6):
    '''Local Learning Regularization for semi-supervised classification.

    partial_labels: (n,) array of integer labels, -1 for unlabeled.

    From "Transductive Classification via Local Learning Regularization"
      by Wu & Scholkopf in 2007.
    '''
    raise NotImplementedError('NYI')

  def classify_harmonic(self, partial_labels, use_CMN=True):
    '''Harmonic function method for semi-supervised classification,
    also known as the Gaussian Mean Fields algorithm.

    partial_labels: (n,) array of integer labels, -1 for unlabeled.
    use_CMN : when True, apply Class Mass Normalization

    From "Semi-Supervised Learning Using Gaussian Fields and Harmonic Functions"
      by Zhu, Ghahramani, and Lafferty in 2003.

    Based on the matlab code at:
      http://pages.cs.wisc.edu/~jerryzhu/pub/harmonic_function.m
    '''
    # prepare labels
    labels = np.array(partial_labels, copy=True)
    unlabeled = labels == -1

    # convert known labels to one-hot encoding
    fl, classes = _onehot(labels[~unlabeled])

    L = self.laplacian(normed=False)
    if ss.issparse(L):
      L = L.tocsr()[unlabeled].toarray()
    else:
      L = L[unlabeled]

    Lul = L[:,~unlabeled]
    Luu = L[:,unlabeled]
    fu = -np.linalg.solve(Luu, Lul.dot(fl))

    if use_CMN:
      scale = (1 + fl.sum(axis=0)) / fu.sum(axis=0)
      fu *= scale

    # assign new labels
    labels[unlabeled] = classes[fu.argmax(axis=1)]
    return labels

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
    S = self.kernelize(kernel).laplacian(normed=True)
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


def _onehot(labels, mask=Ellipsis):
  classes = np.unique(labels[mask])
  onehot = np.zeros((len(labels), len(classes)), dtype=int)
  for idx, label in enumerate(classes):
    onehot[labels==label, idx] = 1
  return onehot, classes
