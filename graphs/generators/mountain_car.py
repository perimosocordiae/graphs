import numpy as np
from graphs import Graph, plot_graph, laplacian_eigenmaps


def plot_mcar_graph(X, W, title, basis_functions=True):
  from matplotlib.pyplot import subplots, figure, show
  from common.viz import irregular_contour, plot

  if not basis_functions:
    return plot_graph(W, X, title=title, fig=figure())

  _, axes = subplots(nrows=2, ncols=2)
  plot_graph(W, X, title=title, ax=axes[0,0])

  emb, vals = laplacian_eigenmaps(W=W, num_vecs=3, return_vals=True)

  x, y = X.T
  for i,ax in enumerate(axes.flat[1:]):
    irregular_contour(x, y, emb[:,i], func=ax.contourf, interp_method='nearest')
    plot(X, marker='k,', ax=ax, title='Basis %d' % (i+1))
  return show


def sample_mcar_trajectories(num_traj, min_length=6, return_traces=False):
  from rl.domains import MountainCar
  from rl.policy import HardCodedPolicy
  from rl.td_agent import PolicyRunner

  # collect data with random hard-coded policies
  domain = MountainCar()
  slopes = np.random.normal(0, 0.01, size=num_traj)
  v0s = np.random.normal(0, 0.005, size=num_traj)
  trajectories = []
  traces = []
  norm = np.array((domain.MAX_POS-domain.MIN_POS,
                   domain.MAX_VEL-domain.MIN_VEL))
  for m,b in zip(slopes, v0s):
    mcar_policy = HardCodedPolicy(lambda s: 0 if s[0]*m + s[1] + b > 0 else 2)
    start = (np.random.uniform(domain.MIN_POS,domain.MAX_POS),
             np.random.uniform(domain.MIN_VEL,domain.MAX_VEL))
    samples = PolicyRunner(mcar_policy).run_episode(domain, start, max_iters=40)
    # normalize
    samples.state /= norm
    samples.next_state /= norm
    traces.append(samples)
    if samples.reward[-1] == 0:
      # Don't include the warp to the final state.
      trajectories.append(samples.state[:-1])
    else:
      trajectories.append(samples.state)

  return trajectories, traces
