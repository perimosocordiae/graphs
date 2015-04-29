import numpy as np

__all__ = ['MobiusStrip', 'FigureEight', 'SCurve']


class ParameterizedShape(object):
  def __init__(self, **param_info):
    for name,(lb,ub,is_monotone) in param_info.items():
      assert lb <= ub, 'Lower bound must be <= upper bound for %s' % name
      assert (bool(is_monotone) == is_monotone
              ), 'monoticity must be boolean for %s' % name
    self.param_info = param_info

  def evaluate(self, **param_values):
    raise NotImplementedError('subclasses must implement this')

  def point_cloud(self, num_points):
    param_values = {}
    for name,(lb,ub,is_monotone) in self.param_info.items():
      if is_monotone:
        vals = np.linspace(lb, ub, num_points)
      else:
        vals = np.random.uniform(lb, ub, size=num_points)
      param_values[name] = vals
    return self.evaluate(**param_values)

  def trajectories(self, num_traj, points_per_traj):
    param_values = {}
    for name,(lb,ub,is_monotone) in self.param_info.items():
      step = float(ub-lb)/points_per_traj
      shape = (num_traj, points_per_traj)
      if is_monotone:
        vals = np.random.normal(loc=step, scale=step/3, size=shape)
      else:
        vals = np.random.normal(loc=0, scale=step, size=shape)
      param_values[name] = np.cumsum(vals, axis=1)
    #TODO: random offsets for starting vals?
    return self.evaluate(**param_values)


class MobiusStrip(ParameterizedShape):
  def __init__(self, radius=1.0, max_width=1.0):
    ParameterizedShape.__init__(self,
                                theta=(0, 2*np.pi, True),
                                width=(-max_width/2, max_width/2, False))
    self.radius = radius

  def evaluate(self, theta=None, width=None):
    tmp = self.radius + width * np.cos(theta/2)
    X = np.empty(theta.shape + (3,))
    X[...,0] = tmp * np.cos(theta)
    X[...,1] = tmp * np.sin(theta)
    X[...,2] = width * np.sin(theta/2)
    return X


class FigureEight(ParameterizedShape):
  def __init__(self, radius=1.0, dimension=2):
    ParameterizedShape.__init__(self,
                                theta=(0, 2*np.pi, True),
                                width=(0, 1, False))  # width is only if it's 3d
    self.radius = radius
    assert dimension in (2,3)
    self.dim = dimension

  def evaluate(self, theta=None, width=None):
    X = np.empty(theta.shape + (self.dim,))
    X[...,0] = self.radius * np.sin(theta)
    # The only difference from a circle is this extra sin(theta) term.
    X[...,1] = X[...,0] * np.cos(theta)
    if self.dim == 3:
      X[...,2] = width
    return X


class SCurve(ParameterizedShape):
  def __init__(self, radius=1.0):
    ParameterizedShape.__init__(self,
                                theta=(-np.pi-1, np.pi+1, True),
                                width=(-1, 1, False))
    self.radius = radius

  def evaluate(self, theta=None, width=None):
    X = np.empty(theta.shape + (3,))
    X[...,0] = np.sin(theta)
    X[...,2] = np.cos(theta)
    X[...,1] = width
    first_half = slice(0, theta.shape[-1]//2)
    X[...,first_half,2] = 2 + -X[...,first_half,2]
    return X
