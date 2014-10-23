import numpy as np

__all__ = ['swiss_roll', 'error_ratio']


def swiss_roll(radians, num_points, radius=1.0,
               theta_noise=0.1, radius_noise=0.01,
               return_theta=False):
  theta = np.linspace(1, radians, num_points)
  if theta_noise > 0:
    theta += np.random.normal(scale=theta_noise, size=theta.shape)
  r = np.sqrt(np.linspace(0, radius*radius, num_points))
  if radius_noise > 0:
    r += np.random.normal(scale=radius_noise, size=r.shape)
  roll = np.empty((num_points, 3))
  roll[:,0] = r * np.sin(theta)
  roll[:,2] = r * np.cos(theta)
  roll[:,1] = np.random.uniform(-1,1,num_points)
  if return_theta:
    return roll, theta
  return roll


def error_ratio(G, GT_points, max_delta_theta=0.1, return_tuple=False):
  theta_edges = GT_points[G.pairs(),0]
  delta_theta = np.abs(np.diff(theta_edges))
  err_edges = np.count_nonzero(delta_theta > max_delta_theta)
  tot_edges = delta_theta.shape[0]
  if return_tuple:
    return err_edges, tot_edges
  return err_edges / float(tot_edges)
