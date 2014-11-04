import numpy as np
import warnings
from scipy.sparse import issparse
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from sklearn.decomposition import KernelPCA

from analysis import shortest_path, laplacian

__all__ = [
    'isomap', 'laplacian_eigenmaps', 'locality_preserving_projections',
    'laplacian_pca'
]


def isomap(G, num_vecs=None, directed=True):
  directed = directed and G.is_directed()
  W = -0.5 * shortest_path(G, directed=directed) ** 2
  return KernelPCA(n_components=num_vecs, kernel='precomputed').fit_transform(W)


def laplacian_eigenmaps(G, num_vecs=None, return_vals=False, val_thresh=1e-8):
  L = laplacian(G, normed=True)
  return _lapeig(L, num_vecs, return_vals, val_thresh)


def locality_preserving_projections(G, coordinates, num_vecs=None):
  X = np.atleast_2d(coordinates)
  L = laplacian(G, normed=True)
  u,s,_ = np.linalg.svd(X.T.dot(X))
  Fplus = np.linalg.pinv(u * np.sqrt(s))
  T = reduce(np.dot,(Fplus,X.T,L,X,Fplus.T))
  L = 0.5*(T+T.T)
  return _lapeig(L, num_vecs, False, 1e-8)


def laplacian_pca(G, coordinates, num_vecs=None, beta=0.5):
  '''Graph-Laplacian PCA (CVPR 2013).
  Assumes coordinates are mean-centered.
  Parameter beta in [0,1], scales how much PCA/LapEig contributes.
  Returns an approximation of input coordinates, ala PCA.'''
  X = np.atleast_2d(coordinates)
  L = laplacian(G, normed=True)
  kernel = X.dot(X.T)
  kernel /= eigsh(kernel, k=1, which='LM', return_eigenvectors=False)
  L /= eigsh(L, k=1, which='LM', return_eigenvectors=False)
  W = (1-beta)*(np.identity(kernel.shape[0]) - kernel) + beta*L
  vals, vecs = eigh(W, eigvals=(0, num_vecs-1), overwrite_a=True)
  return X.T.dot(vecs).dot(vecs.T).T


def _lapeig(L, num_vecs, return_vals, val_thresh):
  if issparse(L):
    # This is a bit of a hack. Make sure we end up with enough eigenvectors.
    k = L.shape[0] - 1 if num_vecs is None else num_vecs + 1
    try:
      # TODO: try using shift-invert mode (sigma=0?) for speed here.
      vals,vecs = eigsh(L, k, which='SM')
    except:
      warnings.warn('Sparse eigsh failed, falling back to dense version')
      vals,vecs = eigh(L.A, overwrite_a=True)
  else:
    vals,vecs = eigh(L, overwrite_a=True)
  # vals not guaranteed to be in sorted order
  idx = np.argsort(vals)
  vecs = vecs.real[:,idx]
  vals = vals.real[idx]
  # discard any with really small eigenvalues
  i = np.searchsorted(vals, val_thresh)
  if num_vecs is None:
    # take all of them
    num_vecs = vals.shape[0] - i
  embedding = vecs[:,i:i+num_vecs]
  if return_vals:
    return embedding, vals[i:i+num_vecs]
  return embedding
