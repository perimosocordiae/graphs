import numpy as np
from matplotlib import pyplot
from matplotlib.axes import mlines, mcolors
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

__all__ = ['plot_graph']


def plot_graph(G, coordinates, undirected=False, unweighted=False, fig=None,
               ax=None, edge_style=None, vertex_style=None, title=None,
               cmap=None):
  X = np.atleast_2d(coordinates)
  assert X.shape[1] in (2,3), 'can only plot graph for 2d or 3d coordinates'
  is_3d = (X.shape[1] == 3)
  if ax is None:
    ax = _get_axis(is_3d, fig)
  edge_kwargs = dict(colors='b', linestyles='-', zorder=1)
  vertex_kwargs = dict(marker='o', c='k', s=20, edgecolor='none', zorder=2)
  if edge_style:
    if isinstance(edge_style, basestring):
      edge_style = _parse_fmt(edge_style, color_key='colors')
    edge_kwargs.update(edge_style)
  if vertex_style:
    if isinstance(vertex_style, basestring):
      vertex_style = _parse_fmt(vertex_style, color_key='c')
    vertex_kwargs.update(vertex_style)
  if not unweighted and G.is_weighted():
    edge_kwargs['array'] = G.edge_weights()
  if not undirected and G.is_directed():
    _directed_edges(G, X, ax, is_3d, edge_kwargs, cmap)
  else:
    _undirected_edges(G, X, ax, is_3d, edge_kwargs, cmap)
  ax.scatter(*X.T, **vertex_kwargs)
  ax.autoscale_view()
  if title:
    ax.set_title(title)
  return pyplot.show


def _parse_fmt(fmt, color_key='colors', ls_key='linestyles',
               marker_key='marker'):
  '''Modified from matplotlib's _process_plot_format function.'''
  try:  # Is fmt just a colorspec?
    color = mcolors.colorConverter.to_rgb(fmt)
    # We need to differentiate grayscale '1.0' from tri_down marker '1'
    try:
      fmtint = str(int(fmt))
    except ValueError:
      return {color_key:color}
    else:
      if fmt != fmtint:
        # user definitely doesn't want tri_down marker
        return {color_key:color}
  except ValueError:
    pass  # No, not just a color.

  result = dict()
  # handle the multi char special cases and strip them from the string
  if fmt.find('--') >= 0:
    result[ls_key] = '--'
    fmt = fmt.replace('--', '')
  if fmt.find('-.') >= 0:
    result[ls_key] = '-.'
    fmt = fmt.replace('-.', '')
  if fmt.find(' ') >= 0:
    result[ls_key] = 'None'
    fmt = fmt.replace(' ', '')

  for c in list(fmt):
    if c in mlines.lineStyles:
      if ls_key in result:
        raise ValueError('Illegal format string; two linestyle symbols')
      result[ls_key] = c
    elif c in mlines.lineMarkers:
      if marker_key in result:
        raise ValueError('Illegal format string; two marker symbols')
      result[marker_key] = c
    elif c in mcolors.colorConverter.colors:
      if color_key in result:
        raise ValueError('Illegal format string; two color symbols')
      result[color_key] = c
    else:
      raise ValueError('Unrecognized character %c in format string' % c)
  return result


def _directed_edges(G, X, ax, is_3d, edge_style, cmap):
  ii, jj = G.pairs().T
  start, stop = X[ii].T, X[jj].T
  dirs = stop - start
  if is_3d:
    x,y,z = start
    dx,dy,dz = dirs
    ax.quiver(x, y, z, dx, dy, dz, cmap=cmap)
  else:
    x,y = start
    dx,dy = dirs
    if hasattr(ax, 'zaxis'):  # Might be on a 3d plot axis.
      z = np.zeros_like(x)
      dz = np.zeros_like(dx)
      ax.quiver(x, y, z, dx, dy, dz, cmap=cmap)
    else:
      args = (x, y, dx, dy)
      ax.quiver(*args, angles='xy', scale_units='xy', scale=1, headwidth=5,
                cmap=cmap)


def _undirected_edges(G, X, ax, is_3d, edge_style, cmap):
  t = X[G.pairs()]
  if is_3d:
    edges = Line3DCollection(t, cmap=cmap, **edge_style)
    ax.add_collection3d(edges)
  else:
    edges = LineCollection(t, cmap=cmap, **edge_style)
    ax.add_collection(edges, autolim=True)


def _get_axis(is_3d, fig):
  if is_3d:
    from mpl_toolkits.mplot3d import Axes3D
    if fig is None:
      fig = pyplot.gcf()
    return Axes3D(fig)
  return pyplot.gca()
