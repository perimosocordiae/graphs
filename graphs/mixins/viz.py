from __future__ import print_function, absolute_import, division
import numpy as np
from matplotlib import pyplot
from matplotlib.axes import mlines, mcolors
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from ..mini_six import zip, zip_longest


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
    assert 0 < X.shape[1] <= 3, 'too many dimensions to plot'
    if X.shape[1] == 1:
      X = np.column_stack((np.arange(X.shape[0]), X))
    if weighted is None:
      weighted = self.is_weighted()
    if directed is None:
      directed = self.is_directed()
    is_3d = (X.shape[1] == 3)
    if ax is None:
      ax = _get_axis(is_3d, fig)
    edge_kwargs = dict(colors='b', linestyles='-', zorder=1)
    vertex_kwargs = dict(marker='o', c='k', s=20, edgecolor='none', zorder=2)
    if edge_style is not None:
      if not isinstance(edge_style, dict):
        edge_style = _parse_fmt(edge_style, color_key='colors')
      edge_kwargs.update(edge_style)
    if vertex_style is not None:
      if not isinstance(vertex_style, dict):
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

  def to_html(self, html_file, directed=None, weighted=None, vertex_ids=None,
              vertex_colors=None, vertex_labels=None, width=900, height=600,
              title=None, svg_border='1px solid black'):
    # input validation
    if weighted is None:
      weighted = self.is_weighted()
    if directed is None:
      directed = self.is_directed()
    if directed:
      raise NotImplementedError('Directed graphs are NYI for HTML output.')
    if (vertex_colors is not None) and (vertex_labels is not None):
      raise ValueError('Supply only one of vertex_colors, vertex_labels')

    # set up vertices
    if vertex_ids is None:
      vertex_ids = np.arange(self.num_vertices())
    elif len(vertex_ids) != self.num_vertices():
      raise ValueError('len(vertex_ids) != num vertices.')

    if vertex_labels is not None:
      vlabels, vcolors = np.unique(vertex_labels, return_inverse=True)
      if len(vcolors) != len(vertex_ids):
        raise ValueError('len(vertex_labels) != num vertices.')
    elif vertex_colors is not None:
      vcolors = np.array(vertex_colors, dtype=float, copy=False)
      if len(vcolors) != len(vertex_ids):
        raise ValueError('len(vertex_colors) != num vertices.')
      vcolors -= vcolors.min()
      vcolors /= vcolors.max()
    else:
      vcolors = []

    node_json = []
    for name, c in zip_longest(vertex_ids, vcolors):
      if c is not None:
        node_json.append('{"id": "%s", "color": %s}' % (name, c))
      else:
        node_json.append('{"id": "%s"}' % name)

    # set up edges
    pairs = self.pairs(directed=directed)
    if weighted:
      weights = self.edge_weights(directed=directed, copy=True).astype(float)
      weights -= weights.min()
      weights /= weights.max()
    else:
      weights = np.zeros(len(pairs)) + 0.5

    edge_json = []
    for (i,j), w in zip(pairs, weights):
      edge_json.append('{"source": "%s", "target": "%s", "weight": %f}' % (
          vertex_ids[i], vertex_ids[j], w))

    # emit self-contained HTML
    if not hasattr(html_file, 'write'):
      fh = open(html_file, 'w')
    else:
      fh = html_file
    print(u'<!DOCTYPE html><meta charset="utf-8"><style>', file=fh)
    print(u'svg { border: %s; }' % svg_border, file=fh)
    if weighted:
      print(u'.links line { stroke-width: 2px; }', file=fh)
    else:
      print(u'.links line { stroke: #000; stroke-width: 2px; }', file=fh)
    print(u'.nodes circle { stroke: #fff; stroke-width: 1px; }', file=fh)
    print(u'</style>', file=fh)
    if title:
      print(u'<h1>%s</h1>' % title, file=fh)
    print(u'<svg width="%d" height="%d"></svg>' % (width, height), file=fh)
    print(u'<script src="https://d3js.org/d3.v4.min.js"></script>', file=fh)
    print(u'<script>', LAYOUT_JS, sep=u'\n', file=fh)
    if vertex_colors is not None:
      print(u'var vcolor=d3.scaleSequential(d3.interpolateViridis);', file=fh)
    elif vertex_labels is not None:
      scale = 'd3.schemeCategory%d' % (10 if len(vlabels) <= 10 else 20)
      print(u'var vcolor = d3.scaleOrdinal(%s);' % scale, file=fh)
    else:
      print(u'function vcolor(){ return "#1776b6"; }', file=fh)
    print(u'var sim=layout_graph({"nodes": [%s], "links": [%s]});</script>' % (
        ',\n'.join(node_json), ',\n'.join(edge_json)), file=fh)
    fh.flush()


def _parse_fmt(fmt, color_key='colors', ls_key='linestyles',
               marker_key='marker'):
  '''Modified from matplotlib's _process_plot_format function.'''
  try:  # Is fmt just a colorspec?
    color = mcolors.colorConverter.to_rgb(fmt)
  except ValueError:
    pass  # No, not just a color.
  else:
    # Either a color or a numeric marker style
    if fmt not in mlines.lineMarkers:
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
      ax.quiver(*args, angles='xy', scale_units='xy', scale=1, cmap=cmap,
                headwidth=3, headlength=3, headaxislength=3,
                alpha=edge_style.get('alpha', 1))


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


LAYOUT_JS = u'''
function layout_graph(graph) {
  var svg = d3.select("svg"),
      width = +svg.attr("width"),
      height = +svg.attr("height");

  var simulation = d3.forceSimulation()
    .force("charge", d3.forceManyBody())
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("link", d3.forceLink()
                     .id(function(d) { return d.id; })
                     .distance(function(d) { return d.weight*20 + 20; })
                     .strength(1));

  function dragstarted(d) {
    if (!d3.event.active) simulation.alphaTarget(0.3).restart();
    d.fx = d.x;
    d.fy = d.y;
  }

  function dragged(d) {
    d.fx = d3.event.x;
    d.fy = d3.event.y;
  }

  function dragended(d) {
    if (!d3.event.active) simulation.alphaTarget(0);
    d.fx = null;
    d.fy = null;
  }

  var ecolor = d3.scaleSequential(d3.interpolateViridis);
  var container = svg.append("g");
  var link = container.append("g").attr("class", "links")
    .selectAll("line").data(graph.links)
    .enter().append("line")
      .attr("stroke", function(d) { return ecolor(d.weight); });

  var node = container.append("g").attr("class", "nodes")
    .selectAll("circle").data(graph.nodes)
    .enter().append("circle")
      .attr("r", 5)
      .attr("fill", function(d) { return vcolor(d.color); })
      .call(d3.drag().on("start", dragstarted)
                     .on("drag", dragged)
                     .on("end", dragended));
  node.append("title").text(function(d) { return d.id; });

  function zoomed() {
    container.attr("transform", d3.event.transform);
    container.selectAll("circle").attr("r", 5/d3.event.transform.k)
  }
  svg.call(d3.zoom().on("zoom", zoomed));

  function ticked() {
    link.attr("x1", function(d) { return d.source.x; })
        .attr("y1", function(d) { return d.source.y; })
        .attr("x2", function(d) { return d.target.x; })
        .attr("y2", function(d) { return d.target.y; });
    node.attr("cx", function(d) { return d.x; })
        .attr("cy", function(d) { return d.y; });
  }
  simulation.nodes(graph.nodes).on("tick", ticked);
  simulation.force("link").links(graph.links);
  return simulation;
}
'''
