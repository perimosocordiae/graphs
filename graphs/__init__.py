'''
Graphs: a library for efficiently manipulating graphs.

 Graph        -- the base class for all graph objects.
 reorder      -- a module for reordering graph vertices.
 construction -- a module for constructing graphs from data.
 generators   -- a module for generating sample datasets.

To create a Graph object, use the static constructors:
 `Graph.from_adj_matrix` or `Graph.from_edge_pairs`.
'''

from base import Graph
import reorder
import construction
import generators
