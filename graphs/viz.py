import numpy as np
from matplotlib import pyplot
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def plot(G, coordinates, undirected=False, unweighted=False, fig=None, ax=None,
         edge_style=None, vertex_style=None):
  X = np.atleast_2d(coordinates)
  assert X.shape[1] in (2,3), 'can only plot graph for 2d or 3d coordinates'
  is_3d = (X.shape[1] == 3)
  if ax is None:
    ax = _get_axis(is_3d, fig)
  edge_kwargs = dict(colors='r', linestyles='-', zorder=1)
  vertex_kwargs = dict(marker='o', c='k', s=20, edgecolor='none', zorder=2)
  if edge_style:
    edge_kwargs.update(edge_style)
  if vertex_style:
    vertex_kwargs.update(vertex_style)
  if not unweighted and G.is_weighted():
    edge_kwargs['array'] = G.edge_weights()
  if not undirected and G.is_directed():
    _directed_edges(G, X, ax, is_3d, edge_kwargs)
  else:
    _undirected_edges(G, X, ax, is_3d, edge_kwargs)
  ax.scatter(*X.T, **vertex_kwargs)
  ax.autoscale_view()
  return ax


def _directed_edges(G, X, ax, is_3d, edge_style):
  start,stop = X[G.pairs()].T
  dirs = stop - start
  if is_3d:
    x,y,z = start
    dx,dy,dz = dirs
    ax.quiver(x, y, z, dx, dy, dz)
  else:
    x,y = start
    dx,dy = dirs
    if hasattr(ax, 'zaxis'):  # Might be on a 3d plot axis.
      z = np.zeros_like(x)
      dz = np.zeros_like(dx)
      ax.quiver(x, y, z, dx, dy, dz)
    else:
      args = (x, y, dx, dy)
      ax.quiver(*args, angles='xy', scale_units='xy', scale=1, headwidth=5)


def _undirected_edges(G, X, ax, is_3d, edge_style):
  t = X[G.pairs()]
  if is_3d:
    edges = Line3DCollection(t, **edge_style)
    ax.add_collection3d(edges)
  else:
    edges = LineCollection(t, **edge_style)
    ax.add_collection(edges, autolim=True)


def _get_axis(is_3d, fig):
  if is_3d:
    from mpl_toolkits.mplot3d import Axes3D
    if fig is None:
      fig = pyplot.gcf()
    return Axes3D(fig)
  return pyplot.gca()
