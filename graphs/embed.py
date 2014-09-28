from sklearn.decomposition import KernelPCA
from analysis import shortest_path


def isomap(G, num_vecs=None, directed=True):
  directed = directed and G.is_directed()
  W = -0.5 * shortest_path(G, directed=directed) ** 2
  return KernelPCA(n_components=num_vecs, kernel='precomputed').fit_transform(W)
