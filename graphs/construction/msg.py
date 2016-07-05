from __future__ import absolute_import, print_function

import numpy as np
from scipy.sparse import issparse
from scipy.sparse.csgraph import dijkstra
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
from graphs import Graph

from ..mini_six import range
from .neighbors import neighbor_graph, nearest_neighbors

__all__ = ['manifold_spanning_graph']


def manifold_spanning_graph(X, embed_dim, num_ccs=1, verbose=False):
  G = neighbor_graph(X, k=1, weighting='binary')
  G.symmetrize(method='max')

  G = grow_trees(X, G, embed_dim, verbose=verbose)

  CC_labels, angle_thresh = join_CCs(X, G, embed_dim, num_ccs=num_ccs,
                                     verbose=verbose)

  adj = G.matrix(dense=True)
  if num_ccs == 1:
    G = flesh_out(X, adj, embed_dim, CC_labels, angle_thresh=angle_thresh,
                  min_shortcircuit=embed_dim+1, verbose=verbose)
  else:
    n, labels = G.connected_components(return_labels=True)
    for i in range(n):
      mask = labels==i
      if verbose:  # pragma: no cover
        print('CC', i, 'has size', np.count_nonzero(mask))
      idx = np.ix_(mask, mask)
      _, tmp_labels = np.unique(CC_labels[mask], return_inverse=True)
      adj[idx] = flesh_out(X[mask], adj[idx], embed_dim, tmp_labels,
                           angle_thresh=angle_thresh,
                           min_shortcircuit=embed_dim+1,
                           verbose=verbose).matrix()
    G = Graph.from_adj_matrix(adj)
  return G


def flesh_out(X, W, embed_dim, CC_labels, dist_mult=2.0, angle_thresh=0.2,
              min_shortcircuit=4, max_degree=5, verbose=False):
  '''Given a connected graph adj matrix (W), add edges to flesh it out.'''
  W = W.astype(bool)
  assert np.all(W == W.T), 'graph given to flesh_out must be symmetric'
  D = pairwise_distances(X, metric='sqeuclidean')

  # compute average edge lengths for each point
  avg_edge_length = np.empty(X.shape[0])
  for i,nbr_mask in enumerate(W):
    avg_edge_length[i] = D[i,nbr_mask].mean()

  # candidate edges must satisfy edge length for at least one end point
  dist_thresh = dist_mult * avg_edge_length
  dist_mask = (D < dist_thresh) | (D < dist_thresh[:,None])
  # candidate edges must connect points >= min_shortcircuit hops away
  hops_mask = np.isinf(dijkstra(W, unweighted=True, limit=min_shortcircuit-1))
  # candidate edges must not already be connected, or in the same initial CC
  CC_mask = CC_labels != CC_labels[:,None]
  candidate_edges = ~W & dist_mask & hops_mask & CC_mask
  if verbose:  # pragma: no cover
    print('before F:', candidate_edges.sum(), 'potentials')

  # calc subspaces
  subspaces, _ = cluster_subspaces(X, embed_dim, CC_labels.max()+1, CC_labels)

  # upper triangular avoids p,q <-> q,p repeats
  ii,jj = np.where(np.triu(candidate_edges))
  # Get angles
  edge_dirs = X[ii] - X[jj]
  ssi = subspaces[CC_labels[ii]]
  ssj = subspaces[CC_labels[jj]]
  F = edge_cluster_angle(edge_dirs, ssi, ssj)

  mask = F < angle_thresh
  edge_ii = ii[mask]
  edge_jj = jj[mask]
  edge_order = np.argsort(F[mask])
  if verbose:  # pragma: no cover
    print('got', len(edge_ii), 'potential edges')
  # Prevent any one node from getting a really high degree
  degree = W.sum(axis=0)
  sorted_edges = np.column_stack((edge_ii, edge_jj))[edge_order]
  for e in sorted_edges:
    if degree[e].max() < max_degree:
      W[e[0],e[1]] = True
      W[e[1],e[0]] = True
      degree[e] += 1
  return Graph.from_adj_matrix(np.where(W, np.sqrt(D), 0))


def grow_trees(X, G, embed_dim, verbose=False):
  dist_thresh = 0
  while True:
    n, labels = G.connected_components(return_labels=True)
    tree_sizes = np.bincount(labels)
    min_tree_size = tree_sizes.min()
    if min_tree_size > embed_dim:
      break
    Dcenter, min_edge_idxs = inter_cluster_distance(X, n, labels)
    pairs = nearest_neighbors(Dcenter, precomputed=True, k=2)  # self + 1 == 2
    ninds = pairs[tree_sizes == min_tree_size]
    meta_edge_lengths = Dcenter[ninds[:,0],ninds[:,1]]
    dist_thresh = max(dist_thresh, np.max(meta_edge_lengths))
    if verbose:  # pragma: no cover
      print(n, 'CCs. dist thresh:', dist_thresh)
    # modify G to connect edges between nearby CCs
    G, num_added = _connect_meta_edges(X, G, None, labels, ninds,
                                       dist_thresh=dist_thresh)[:2]
    assert num_added > 0
  return G


def join_CCs(X, G, embed_dim, num_ccs=1, max_angle=0.3, verbose=False):
  n, labels = G.connected_components(return_labels=True)
  # compute linear subspaces for each connected component
  #  (assumed to be local+linear)
  CC_planes, _ = cluster_subspaces(X, embed_dim, n, labels)
  CC_labels = labels  # keep around the original labels that go with CC_planes
  angle_thresh = 0.1
  while n > num_ccs:
    # compute the distance between all clusters
    #   (by finding the distance between the closest 2 member points)
    Dcenter, min_edge_idxs = inter_cluster_distance(X, n, labels)
    # Find "meta-edges" between clusters (k=1)
    ninds = nearest_neighbors(Dcenter, precomputed=True, k=2)  # self + 1 == 2
    meta_edge_lengths = Dcenter[ninds[:,0],ninds[:,1]]
    dist_thresh = np.median(meta_edge_lengths)
    if verbose:  # pragma: no cover
      print(n, 'CCs')
    # convert ninds to CC_ninds (back to the CC_labels space, via W-space)
    CC_ninds = CC_labels[min_edge_idxs[ninds[:,0],ninds[:,1]]]
    # modify G to connect edges between nearby CCs
    while True:
      if verbose:  # pragma: no cover
        print('DT:', dist_thresh, 'AT:', angle_thresh)
      G, num_added, minD, minF = _connect_meta_edges(
          X, G, CC_planes, CC_labels, CC_ninds,
          dist_thresh=dist_thresh, angle_thresh=angle_thresh)
      if num_added > 0:
        break
      elif angle_thresh < minF <= max_angle:
        angle_thresh = minF
      elif dist_thresh < minD:
        if np.isinf(minD):
          max_angle += 0.1  # XXX: hack
          angle_thresh = min(minF, max_angle)
          if verbose:  # pragma: no cover
            print('Increasing max_angle to', max_angle)
        else:
          dist_thresh = minD
      else:
        raise AssertionError("Impossible state: can't increase dist_thresh "
                             "enough to make a connection")

    # recalc CCs and repeat (keeping the original CC_planes!)
    #  until there's only one left.
    n, labels = G.connected_components(return_labels=True)
  return CC_labels, angle_thresh


def _connect_meta_edges(X, G, CC_planes, CC_labels, CC_ninds,
                        dist_thresh=0.1, angle_thresh=0.1):
  # For each "meta-edge" (from cluster P to Q)
  min_F = 1.0
  min_D = np.inf
  W = G.matrix(dense=True, coo=True)
  # COO format only supports sum_duplicates since August 2014.
  if issparse(W) and not hasattr(W, 'sum_duplicates'):
    W = W.toarray()
  to_add = []
  for p,q in CC_ninds:
    ii, = np.where(CC_labels==p)
    jj, = np.where(CC_labels==q)
    # Compute the distance between all points in P and Q
    Dc = pairwise_distances(X[ii], X[jj], metric='sqeuclidean')
    Dmask = Dc <= dist_thresh
    if CC_planes is not None and np.any(Dmask):
      # Compute the direction of all potential edges between P and Q
      edge_dir = (X[ii,None] - X[jj]).reshape((-1,X.shape[1]))
      # Find the maximum angle between the edge and its two endpoint clusters
      F = edge_cluster_angle(edge_dir, CC_planes[p], CC_planes[q]
                             ).reshape(Dc.shape)
      Fmask = F <= angle_thresh
      min_F = min(min_F, F[Dmask].min())
      if np.any(Fmask):
        min_D = min(min_D, Dc[Fmask].min())
    else:
      F = 0
      Fmask = True
      min_D = min(min_D, Dc.min())
    # add P-Q edges, only if distance (Dc) is small AND max angle (F) is small
    pq_edges = np.argwhere(Dmask & Fmask)
    # Scale distance to the [0,1] range.
    # F is already in [0,1], so no scaling needed.
    Dc -= Dc.min()
    Dc /= Dc.max()
    # ensure that each point we're connecting doesn't already connect P-Q.
    combined = F + Dc
    while pq_edges.size > 0:
      # select and add the edge that minimizes Dc + F
      ei,ej = pq_edges[np.argmin(combined[pq_edges[:,0],pq_edges[:,1]])]
      to_add.append((ii[ei], jj[ej]))
      # remove any additional candidate edges from/to the added edge's endpoints
      pq_edges = pq_edges[(pq_edges[:,0] != ei) & (pq_edges[:,1] != ej)]
  if to_add:
    ii,jj = np.transpose(to_add)
    if issparse(W):
      # Yep, this is nasty
      assert W.format == 'coo', 'Only COO matrix is supported for MSG'
      W.row = np.hstack((W.row, ii, jj))
      W.col = np.hstack((W.col, jj, ii))
      W.data = np.append(W.data, np.ones(len(ii)+len(jj), dtype=W.data.dtype))
      W.sum_duplicates()
    else:
      W[ii,jj] = 1
      W[jj,ii] = 1
  return Graph.from_adj_matrix(W), len(to_add), min_D, min_F


def edge_cluster_angle(edge_dirs, subspaces1, subspaces2):
  '''edge_dirs is a (n,D) matrix of edge vectors.
  subspaces are (n,D,d) or (D,d) matrices of normalized orthogonal subspaces.
  Result is an n-length array of angles.'''
  QG = edge_dirs / np.linalg.norm(edge_dirs, ord=2, axis=1)[:,None]
  X1 = np.einsum('...ij,...i->...j', subspaces1, QG)
  X2 = np.einsum('...ij,...i->...j', subspaces2, QG)
  # TODO: check the math on this for more cases
  # angles = np.maximum(1-np.sum(X1**2, axis=1), 1-np.sum(X2**2, axis=1))
  C1 = np.linalg.svd(X1[:,:,None], compute_uv=False)
  C2 = np.linalg.svd(X2[:,:,None], compute_uv=False)
  angles = np.maximum(1-C1**2, 1-C2**2)[:,0]
  angles[np.isnan(angles)] = 0.0  # nan when edge length == 0
  return angles


def cluster_subspaces(X, subspace_dim, num_clusters, cluster_labels):
  means = np.empty((num_clusters, X.shape[1]))
  subspaces = np.empty((num_clusters, X.shape[1], subspace_dim))
  for i in range(num_clusters):
    CC = X[cluster_labels==i]
    means[i] = CC.mean(axis=0)
    pca = PCA(n_components=subspace_dim).fit(CC)
    subspaces[i] = pca.components_.T
  return subspaces, means


def _inter_cluster_distance(X, num_clusters, cluster_labels):
  # compute shortest distances between clusters
  Dx = pairwise_distances(X, metric='sqeuclidean')
  Dc = np.zeros((num_clusters,num_clusters))
  edges = np.zeros((num_clusters,num_clusters,2), dtype=int)
  index_array = np.arange(X.shape[0])
  masks = cluster_labels == np.arange(num_clusters)[:,None]
  indices = [index_array[m] for m in masks]
  for i in range(num_clusters-1):
    inds = indices[i]
    dists = Dx[masks[i]].T
    for j in range(i+1, num_clusters):
      m2 = masks[j]
      d = dists[m2].T
      min_idx = np.argmin(d)
      min_val = d.flat[min_idx]
      r,c = np.unravel_index(min_idx, d.shape)
      edges[i,j] = (inds[r], indices[j][c])
      edges[j,i] = edges[i,j]
      Dc[i,j] = min_val
      Dc[j,i] = min_val
  return Dc, edges

try:
  import pyximport
  pyximport.install(setup_args={'include_dirs': np.get_include()})
  from ._fast_paths import inter_cluster_distance
except ImportError:
  inter_cluster_distance = _inter_cluster_distance
