'''
Graphs: a library for efficiently manipulating graphs.

 Graph        -- the base class for all graph objects.
 construction -- a module for constructing graphs from data.
 generators   -- a module for generating graphs with desired properties.
 datasets     -- a module providing sample datasets.
 reorder      -- a module for reordering graph vertices.

To create a Graph object, use the static constructors:
 `Graph.from_adj_matrix` or `Graph.from_edge_pairs`.
'''
from __future__ import absolute_import

from ._version import __version__
from .base import Graph
from . import construction, generators, datasets, reorder
