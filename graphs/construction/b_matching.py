import numpy as np


def b_matching_jebara(D, k, max_iter=999):
    """http://www.cs.columbia.edu/~jebara/code/loopy/"""
    N = D.shape[0]
    D = np.exp(-D)  # convert to similarity
    ii = np.arange(N)

    def _find_mininds(work_vecs, k):
        inds = np.repeat(np.arange(k)[None], N, axis=0)
        mininds = np.argmin(work_vecs[inds.T,ii], axis=0)
        for j in xrange(k,N):
            cond = work_vecs[j] > work_vecs[inds[ii,mininds],ii]
            inds[cond,mininds[cond]] = j
            mininds = np.argmin(work_vecs[inds.T,ii], axis=0)
        return inds, mininds

    def _update(work_vecs, transposed, out):
        inds, mininds = _find_mininds(work_vecs, k+1)
        true_kth = np.argpartition(work_vecs[inds,ii[:,None]], 1)[:,1]
        true_vecs = work_vecs[inds[ii,true_kth],ii]
        if transposed:
            in_top_k = work_vecs.T >= true_vecs[:,None]
        else:
            in_top_k = work_vecs >= true_vecs
        jj = np.nonzero(in_top_k)[0]
        out[in_top_k] = D[in_top_k] / work_vecs[inds[jj,mininds[jj]],jj]
        out[~in_top_k] = D[~in_top_k] / true_vecs[np.nonzero(~in_top_k)[0]]

    # run the iterative matching
    P = np.zeros_like(D, dtype=bool)
    alpha = np.ones_like(D)
    beta = np.ones_like(D)
    found_answer = 0.0
    for it in xrange(max_iter):
        # update beta from alpha
        _update(D * alpha, True, beta)
        work_vecs = D.T * beta
        # compute beliefs
        P[:] = False
        inds = _find_mininds(work_vecs, k)[0]
        P[ii,inds.T] = True
        # check for convergence
        num_invalid_edges = (np.count_nonzero(P.sum(axis=0) != k) +
                             np.count_nonzero(P.sum(axis=1) != k))
        if it >= N and num_invalid_edges == 0:
            found_answer += 1
        else:
            found_answer -= 0.1
        if found_answer > 0:
            print "Converged in %d steps" % it
            break
        # update alpha from beta
        _update(work_vecs, False, alpha)
    else:
        print "Didn't converge in %d steps" % max_iter
    return P


def b_matching(D, k, all_ranks=False, epsilon=-0.5, max_iter=5000):
    '''
    "Belief-Propagation for Weighted b-Matchings on Arbitrary Graphs
    and its Relation to Linear Programs with Integer Solutions"
    Bayati et al.

    Finds the minimal weight perfect b-matching using min-sum loopy-BP.
    This is a *symmetric* alternative to a kNN matrix.

    @param D pairwise distance matrix, only symmetric matrices currently
    @param k number of neighbors
    @param all_ranks returns the full nearness ranking of all neighbors
    @param epsilon a converges when k messages (per sample) are below eps.
    @param max_iter cap on the number of iterations of message passing

    @return binary adjacency matrix of neighbors
    '''

    n = D.shape[0]
    assert D.shape[1] == n, 'Input distances matrix must be square'
    # TODO: add non-symmetric matrix support, maybe something like...
    #dim_diff = D.shape[0] - D.shape[1]
    #if dim_diff > 0: #tall
    #    D = np.hstack((D, np.ones((D.shape[0],dim_diff))*float('inf')))
    #elif dim_diff < 0: #short
    #    D = np.vstack((D,np.ones((abs(dim_diff),D.shape[1]))*float('inf')))

    # TODO: In theory, epsilon should always be zero, but in practice that often
    # leads to a bad convergence (to a non-symmetric matrix).
    # This ain't right and someone should figure it out.

    msg = D
    for t_step in xrange(max_iter):
        sorted_msg = np.partition(msg, (k-1,k), axis=0)
        d1 = sorted_msg[k-1,None].T
        d2 = sorted_msg[k,None].T
        mask = msg.T <= d1
        msg = D - np.where(mask, d2, d1)
        if np.count_nonzero(msg < epsilon) == k*n:
            print "Converged in %d steps" % t_step
            break
    else:
        print "Didn't converge in %d steps" % max_iter

    sorted_indices = np.argsort(np.argsort(msg))
    binary_indices = sorted_indices < k
    if all_ranks:
        return binary_indices, sorted_indices
    return binary_indices


def demo():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.axes3d import Axes3D
    from graphs import Graph, plot_graph, connected_components
    from common.neighborhood import neighbor_graph
    from common.synthetic_data import swiss_roll
    from common.util import Timer
    from sklearn.metrics.pairwise import pairwise_distances

    n = 50
    pts = swiss_roll(np.linspace(0,5,n))
    #pts = np.asarray([[0,1,],[1,1],[0,0,],[1,0],[3,3],[4,4]])

    dists = pairwise_distances(pts)
    # search for k with one CC
    for k in xrange(3, 10):
        nn_adj = neighbor_graph(dists, k=k, precomputed=True)
        n = connected_components(nn_adj, return_labels=False)
        if n == 1:
            print 'Using k =', k
            break
    else:
        assert False, 'k too small to make one connected component'

    with Timer('b-matching'):
        bm_adj = b_matching(dists, k+1)
    with Timer('Jebara'):
        bmj_adj = b_matching_jebara(dists, k+1)

    print 'BM and BMJ differ by %d edges' % np.count_nonzero(bm_adj != bmj_adj)

    knn_ax, bm1_ax, bm2_ax = plt.subplots(ncols=3,
                                          subplot_kw=dict(projection='3d'))[1]
    plot_graph(nn_adj,pts,title='kNN',ax=knn_ax)
    plot_graph(Graph.from_adj_matrix(bm_adj), pts, title='b-matching',ax=bm1_ax)
    plot_graph(Graph.from_adj_matrix(bmj_adj), pts,
               title='Jebara b-matching',ax=bm2_ax)
    plt.show()


if __name__ == '__main__':
    demo()
