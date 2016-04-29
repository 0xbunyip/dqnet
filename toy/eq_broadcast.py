import numpy as np
import theano
import theano.tensor as T

np.random.seed(123)

m = 8
n = 4

action = T.col(dtype = 'uint8', name = 'action')
values = T.matrix(dtype = 'float32', name = 'values')

q = values[T.arange(m), action.reshape((-1,))]
print theano.printing.debugprint(q)
f = theano.function([action, values], q)


x = T.eq(T.arange(n).reshape((1, -1)), action.reshape((-1, 1)))
y = T.sum(values * x, axis = 1)
print theano.printing.debugprint(y)
g = theano.function([action, values], y)

v = -np.random.rand(m, n).astype(dtype = np.float32)
a = np.random.randint(0, n, size = (m, 1)).astype(dtype = np.uint8)
# print v
# print a
ff = f(a, v)
gg = g(a, v)
# print ff
# print gg
print np.allclose(ff, gg)