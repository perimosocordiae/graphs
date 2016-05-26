import numpy as np
import warnings
from scipy.sparse import issparse
from scipy.sparse.linalg import eigsh
from scipy.linalg import eig, eigh
from sklearn.decomposition import KernelPCA


class EmbedMixin(object):

  def isomap(self, num_dims=None, directed=True):
    '''Isomap embedding.'''
    directed = directed and self.is_directed()
    W = -0.5 * self.shortest_path(directed=directed) ** 2
    kpca = KernelPCA(n_components=num_dims, kernel='precomputed')
    return kpca.fit_transform(W)

  def laplacian_eigenmaps(self, num_dims=None, val_thresh=1e-8):
    '''Laplacian Eigenmaps embedding.'''
    L = self.laplacian(normed=True)
    return _null_space(L, num_dims, val_thresh, overwrite=True)

  def locality_preserving_projections(self, coordinates, num_dims=None):
    '''Locality Preserving Projections (LPP, linearized Laplacian Eigenmaps).'''
    X = np.atleast_2d(coordinates)  # n x d
    L = self.laplacian(normed=True)  # n x n
    u,s,_ = np.linalg.svd(X.T.dot(X))
    Fplus = np.linalg.pinv(u * np.sqrt(s))  # d x d
    n, d = X.shape
    if n >= d:  # optimized order: F(X'LX)F'
      T = Fplus.dot(X.T.dot(L).dot(X)).dot(Fplus.T)
    else:  # optimized order: (FX')L(XF')
      T = Fplus.dot(X.T).dot(L).dot(X.dot(Fplus.T))
    L = 0.5*(T+T.T)
    return _null_space(L, num_vecs=num_dims, overwrite=True)

  def locally_linear_embedding(self, num_dims=None):
    '''Locally Linear Embedding (LLE).
    Note: may need to call barycenter_edge_weights() before this!
    '''
    W = self.matrix()
    # compute M = (I-W)'(I-W)
    M = W.T.dot(W) - W.T - W
    if issparse(M):
      M = M.toarray()
    M.flat[::M.shape[0] + 1] += 1
    return _null_space(M, num_vecs=num_dims, overwrite=True)

  def neighborhood_preserving_embedding(self, X, num_dims=None, reweight=True):
    '''Neighborhood Preserving Embedding (NPE, linearized LLE).'''
    if reweight:
      W = self.barycenter_edge_weights(X).matrix()
    else:
      W = self.matrix()
    # compute M = (I-W)'(I-W) as in LLE
    M = W.T.dot(W) - W.T - W
    if issparse(M):
      M = M.toarray()
    M.flat[::M.shape[0] + 1] += 1
    # solve generalized eig problem: X'MXa = \lambda X'Xa
    vals, vecs = eig(X.T.dot(M).dot(X), X.T.dot(X), overwrite_a=True,
                     overwrite_b=True)
    if num_dims is None:
      return vecs
    return vecs[:,:num_dims]

  def laplacian_pca(self, coordinates, num_dims=None, beta=0.5):
    '''Graph-Laplacian PCA (CVPR 2013).
    coordinates : (n,d) array-like, assumed to be mean-centered.
    beta : float in [0,1], scales how much PCA/LapEig contributes.
    Returns an approximation of input coordinates, ala PCA.'''
    X = np.atleast_2d(coordinates)
    L = self.laplacian(normed=True)
    kernel = X.dot(X.T)
    kernel /= eigsh(kernel, k=1, which='LM', return_eigenvectors=False)
    L /= eigsh(L, k=1, which='LM', return_eigenvectors=False)
    W = (1-beta)*(np.identity(kernel.shape[0]) - kernel) + beta*L
    if num_dims is None:
      vals, vecs = np.linalg.eigh(W)
    else:
      vals, vecs = eigh(W, eigvals=(0, num_dims-1), overwrite_a=True)
    return X.T.dot(vecs).dot(vecs.T).T

  def layout_circle(self):
    '''Position vertices evenly around a circle.'''
    n = self.num_vertices()
    t = np.linspace(0, 2*np.pi, n+1)[:n]
    return np.column_stack((np.cos(t), np.sin(t)))

  def layout_spring(self, num_dims=2, spring_constant=None, iterations=50,
                    initial_temp=0.1, initial_layout=None):
    '''Position vertices using the Fruchterman-Reingold (spring) algorithm.

    num_dims : int (default=2)
       Number of dimensions to embed vertices in.

    spring_constant : float (default=None)
       Optimal distance between nodes.  If None the distance is set to
       1/sqrt(n) where n is the number of nodes.  Increase this value
       to move nodes farther apart.

    iterations : int (default=50)
       Number of iterations of spring-force relaxation

    initial_temp : float (default=0.1)
       Largest step-size allowed in the dynamics, decays linearly.
       Must be positive, should probably be less than 1.

    initial_layout : array-like of shape (n, num_dims)
       If provided, serves as the initial placement of vertex coordinates.
    '''
    if initial_layout is None:
      X = np.random.random((self.num_vertices(), num_dims))
    else:
      X = np.array(initial_layout, dtype=float, copy=True)
      assert X.shape == (self.num_vertices(), num_dims)
    if spring_constant is None:
      # default to sqrt(area_of_viewport / num_vertices)
      spring_constant = X.shape[0] ** -0.5
    S = self.matrix(csr=True, csc=True, coo=True)
    S.data[:] = 1. / S.data  # Convert to similarity
    ii,jj = S.nonzero()  # cache nonzero indices
    # simple cooling scheme, linearly steps down
    cooling_scheme = np.linspace(initial_temp, 0, iterations+2)[:-2]
    # this is still O(V^2)
    # could use multilevel methods to speed this up significantly
    for t in cooling_scheme:
      delta = X[:,None] - X[None]
      distance = _bounded_norm(delta, 1e-8)
      # repulsion from all vertices
      force = spring_constant**2 / distance
      # attraction from connected vertices
      force[ii,jj] -= S.data * distance[ii,jj]**2 / spring_constant
      displacement = np.einsum('ijk,ij->ik', delta, force)
      # update positions
      length = _bounded_norm(displacement, 1e-2)
      X += displacement * t / length[:,None]
    return X


def _null_space(X, num_vecs=None, val_thresh=1e-8, overwrite=False):
  if issparse(X):
    # This is a bit of a hack. Make sure we end up with enough eigenvectors.
    k = X.shape[0] - 1 if num_vecs is None else num_vecs + 1
    try:
      # TODO: try using shift-invert mode (sigma=0?) for speed here.
      vals,vecs = eigsh(X, k, which='SM')
    except:
      warnings.warn('Sparse eigsh failed, falling back to dense version')
      X = X.toarray()
      overwrite = True
  if not issparse(X):
    vals,vecs = eigh(X, overwrite_a=overwrite)
  # vals are not guaranteed to be in sorted order
  idx = np.argsort(vals)
  vecs = vecs.real[:,idx]
  vals = vals.real[idx]
  # discard any with really small eigenvalues
  i = np.searchsorted(vals, val_thresh)
  if num_vecs is None:
    # take all of them
    num_vecs = vals.shape[0] - i
  return vecs[:,i:i+num_vecs]


def _bounded_norm(X, min_length):
  length = np.linalg.norm(X, ord=2, axis=-1)
  np.maximum(length, min_length, out=length)
  return length
