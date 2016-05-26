import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import griddata

__all__ = ['plot_mcar_basis', 'sample_mcar_trajectories']


def plot_mcar_basis(G, X, title='Mountain Car graph'):
  _, axes = plt.subplots(nrows=2, ncols=2)
  G.plot(X, title=title, ax=axes[0,0])

  emb = G.laplacian_eigenmaps(num_dims=3)

  x, y = X.T
  # Set up grids for a contour plot
  x_range = (x.min(), x.max())
  y_range = (y.min(), y.max())
  pad_x = 0.05 * -np.subtract.reduce(x_range)
  pad_y = 0.05 * -np.subtract.reduce(y_range)
  grid_x = np.linspace(x_range[0] - pad_x, x_range[1] + pad_x, 100)
  grid_y = np.linspace(y_range[0] - pad_y, y_range[1] + pad_y, 100)
  for i,(ax,z) in enumerate(zip(axes.flat[1:], emb.T)):
    grid_z = griddata((x, y), z, (grid_x[None], grid_y[:,None]),
                      method='nearest')
    ax.contourf(grid_x, grid_y, grid_z)
    ax.plot(x, y, 'k,')
    ax.set_title('Basis %d' % (i+1))
  return plt.show


def sample_mcar_trajectories(num_traj):
  # collect data with random hard-coded policies
  domain = MountainCar()
  slopes = np.random.normal(0, 0.01, size=num_traj)
  v0s = np.random.normal(0, 0.005, size=num_traj)
  trajectories = []
  traces = []
  norm = np.array((domain.MAX_POS-domain.MIN_POS,
                   domain.MAX_VEL-domain.MIN_VEL))
  for m,b in zip(slopes, v0s):
    mcar_policy = lambda s: 0 if s[0]*m + s[1] + b > 0 else 2
    start = (np.random.uniform(domain.MIN_POS,domain.MAX_POS),
             np.random.uniform(domain.MIN_VEL,domain.MAX_VEL))
    samples = _run_episode(mcar_policy, domain, start, max_iters=40)
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


def _run_episode(policy_action, domain, state, max_iters=1e100):
  action = policy_action(state)
  samples = []
  while not domain.finished(state):
    # get new state and action
    new_state = domain.take_action(state, action)
    new_action = policy_action(new_state)
    # update histories
    reward = domain.reward_for(state)
    samples.append((state, action, reward, new_state, new_action))
    if len(samples) >= max_iters:
      break
    state = new_state
    action = new_action
  ds = len(state)
  names = ('state','action','reward','next_state','next_action')
  formats = (('f',(ds,)),int,float,('f',(ds,)),int)
  dtype = dict(names=names, formats=formats)
  return np.array(samples, dtype).view(np.recarray)


class MountainCar(object):
  # directions: fwd neu rev
  action_dirs = [1, 0, -1]
  NUM_ACTIONS = 3

  GOAL_POS = 0.5
  DT = 0.001

  MIN_POS = -1.2
  MAX_POS = 0.5
  MIN_VEL = -0.07
  MAX_VEL = 0.07

  def __init__(self, gravity=-0.0025):
    self.gravity = gravity

  def reward_for(self, state):
    return 0 if state[0] >= MountainCar.GOAL_POS else -1

  def finished(self, state):
    return self.reward_for(state) == 0

  def take_action(self, state, action):
    p,v = state
    a = MountainCar.action_dirs[action]
    new_v = v + (MountainCar.DT*a) + (self.gravity*np.cos(3*p))
    new_v = min(MountainCar.MAX_VEL, max(MountainCar.MIN_VEL, new_v))
    new_p = p + new_v
    if new_p < MountainCar.MIN_POS:
      new_p = MountainCar.MIN_POS
      new_v = 0
    elif new_p > MountainCar.MAX_POS:
      new_p = MountainCar.MAX_POS
      new_v = 0
    return new_p, new_v
