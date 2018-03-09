from __future__ import print_function
import numpy as np
from matplotlib import pyplot as plt
from graphs.construction import neighbor_graph

if hasattr(__builtins__, 'raw_input'):
  input = raw_input


def main():
  print("Select coordinates for graph vertices:")
  plt.plot([])
  coords = np.array(plt.ginput(n=-1, timeout=-1))

  k = int(input("Number of nearest neighbors: "))
  g = neighbor_graph(coords, k=k)

  print("Resulting graph:")
  g.plot(coords, vertex_style='ro')()

if __name__ == '__main__':
  main()
