import unittest
# from graphs import analysis

if __name__ == '__main__':
  unittest.main()


  # def test_num_connected_components(self):
  #   self.assertEqual(2, self.corr.num_connected_components())
  #   p = numpy.vstack((self.p, [2,3]))
  #   c2 = graph.Graph(pairs=p)
  #   self.assertEqual(1, c2.num_connected_components())
  #   p = numpy.array([[0,1],[2,3],[4,5]])
  #   c3 = graph.Graph(pairs=p)
  #   self.assertEqual(3, c3.num_connected_components())

  # def test_greedy_coloring(self):
  #   expected = numpy.array([1,2,3,1,2])
  #   S = self.corr.symmetrize(overwrite=False)
  #   assert_array_equal(S.greedy_coloring(), expected)

  # def test_dist_from(self):
  #   p2 = numpy.array([[0,1],[1,2],[1,2],[3,4]])
  #   c2 = graph.Graph(pairs=p2)
  #   d1 = c2.dist_from(self.corr)
  #   d2 = self.corr.dist_from(c2)
  #   self.assertEqual(d1,d2)
  #   self.assertEqual(d1, 1/6.)  # actually 2/12, but who's counting?

  # def test_warp(self):
  #   A = numpy.arange(4)
  #   # Test trivial matching (along the diagonal)
  #   p = numpy.array([[0,0],[1,1],[2,2],[3,3]])
  #   corr = graph.Graph(pairs=p)
  #   assert_array_equal(corr.warp(A, XtoY=True), A)
  #   assert_array_equal(corr.warp(A, XtoY=False), A)
  #   # Test a slightly more complex example
  #   p = numpy.array([[0,0],[1,0],[2,1],[3,3]])
  #   corr = graph.Graph(pairs=p)
  #   assert_array_equal(corr.warp(A, XtoY=True), numpy.array([0,0,1,3]))
  #   assert_array_equal(corr.warp(A, XtoY=False), numpy.array([0,2,3,3]))

  # def test_ave_laplacian(self):
  #   W = numpy.array([[0,1,2],[1,0,0],[2,0,0]])
  #   g = graph.Graph(matrix=W)
  #   expected = numpy.array([[1,-0.5,0],[-0.5,1,0],[0,0,1]])
  #   assert_array_almost_equal(g.ave_laplacian(), expected)

  # def test_directed_laplacian(self):
  #   W = numpy.array([[1,0,0,0,1],
  #                    [0,1,0,0,0],
  #                    [0,1,1,1,0],
  #                    [0,0,0,1,0],
  #                    [0,0,1,0,1]])
  #   g = graph.Graph(matrix=W)
  #   expected = numpy.array([
  #       [0.0019802,0,0,0,-0.0009901],
  #       [0,0,-0.00146318,0,0],
  #       [0,-0.00146318,0.00585271,-0.00146318,-0.00196059],
  #       [0,0,-0.00146318,0,0],
  #       [-0.0009901,0,-0.00196059,0,0.00392118]])
  #   L = g.directed_laplacian()
  #   assert_array_almost_equal(L, expected)
  #   # test case where some D are zero
  #   D = W.sum(axis=1)
  #   D[1] = 0
  #   expected[1,1] = 0.00489709
  #   assert_array_almost_equal(g.directed_laplacian(D), expected)
