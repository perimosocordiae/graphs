# Graphs

A library for Graph-Based Learning in Python.

Provides several types of graphs container objects,
with associated visualization, analysis, and embedding functions.

## Requirements

Requires recent versions of:

  * numpy
  * scipy
  * scikit-learn
  * matplotlib

Testing requires:

  * nose
  * nose-cov

## Usage example

```python
from graphs.construction import random_graph

G = random_graph([2,3,1,3,2,1,2])

print G.num_vertices()  # 7
print G.num_edges()     # 14

G.symmetrize(method='max')
X = G.isomap(num_vecs=2)

G.plot(X, directed=False, weighted=False, title='isomap embedding')()
```
