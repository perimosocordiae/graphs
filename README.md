# Graphs

[![PyPI version](https://badge.fury.io/py/graphs.svg)](http://badge.fury.io/py/graphs)
[![Build Status](https://travis-ci.org/all-umass/graphs.svg?branch=master)](https://travis-ci.org/all-umass/graphs)
[![Coverage Status](https://coveralls.io/repos/all-umass/graphs/badge.svg?branch=master&service=github)](https://coveralls.io/github/all-umass/graphs?branch=master)

A library for graph-based learning in Python.

Provides several types of graph container objects,
with a unified API for visualization, analysis, transformation,
and embedding.

## Usage example

```python
from graphs.generators import random_graph

G = random_graph([2,3,1,3,2,1,2])

print G.num_vertices()  # 7
print G.num_edges()     # 14

G.symmetrize(method='max')
X = G.isomap(num_dims=2)

G.plot(X, title='isomap embedding')()
```

## Requirements

Requires recent versions of:

  * numpy
  * scipy
  * scikit-learn
  * matplotlib
  * Cython

Optional dependencies:

  * python-igraph
  * graphtool
  * networkx

Testing requires:

  * nose
  * nose-cov

Run the test suite:

```
./run_tests.sh
```
