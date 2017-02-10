import numpy

a = numpy.zeros(shape=(3,4))
a[0] = [2, 0, 4, 2]
a[1] = [-1, 3, -2, 2]
a[2] = [1, -3, 2, -2]
r = numpy.linalg.matrix_rank(a)
print(r)
