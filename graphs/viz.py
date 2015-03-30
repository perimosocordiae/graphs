import numpy as np
from matplotlib import pyplot
from matplotlib.axes import mlines, mcolors
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection


class VizMixin(object):

  def plot(self, coordinates, directed=None, weighted=None, fig='current',
           ax=None, edge_style=None, vertex_style=None, title=None, cmap=None):
    '''Plot the graph using matplotlib in 2 or 3 dimensions.
    coordinates : (n,2) or (n,3) array of vertex coordinates
    directed : if True, edges have arrows indicating direction. Defaults to
                the result of self.is_directed()
    weighted : if True, edges are colored by their weight. Defaults to the
                result of self.is_weighted()
    fig : a matplotlib Figure to use, or one of {new,current}. Defaults to
          'current', which will call gcf(). Only used when ax=None.
    ax : a matplotlib Axes to use. Defaults to gca()
    edge_style : string or dict of styles for edges. Defaults to 'b-'
    vertex_style : string or dict of styles for vertices. Defaults to 'ko'
    title : string to display as the plot title
    cmap : a matplotlib Colormap to use for edge weight coloring
    '''
    X = np.atleast_2d(coordinates)
    assert X.shape[1] in (2,3), 'can only plot graph for 2d or 3d coordinates'
    if weighted is None:
      weighted = self.is_weighted()
    if directed is None:
      directed = self.is_directed()
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
    if weighted and self.is_weighted():
      edge_kwargs['array'] = self.edge_weights()
    if directed and self.is_directed():
      _directed_edges(self, X, ax, is_3d, edge_kwargs, cmap)
    else:
      _undirected_edges(self, X, ax, is_3d, edge_kwargs, cmap)
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
  except ValueError:
    pass  # No, not just a color.
  else:
    # Yes, just a color (or maybe tri_down marker '1')
    if fmt != '1':
      return {color_key:color}

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
  if fig is 'current':
    fig = pyplot.gcf()
  elif fig is 'new':
    fig = pyplot.figure()
  # Only make a new Axes3D if we need to.
  if is_3d and not (fig.axes and hasattr(fig.gca(), 'zaxis')):
    from mpl_toolkits.mplot3d import Axes3D
    return Axes3D(fig)
  return fig.gca()
