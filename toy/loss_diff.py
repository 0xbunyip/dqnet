import numpy as np
import theano
import theano.tensor as T

error = T.dcol()
q = T.minimum(abs(error), 1.0)
l = abs(error) - q
loss = T.sum(0.5 * q ** 2 + l)
d = theano.grad(loss, error)
f = theano.function([error], d)

e = np.arange(-2, 2.01, 0.25).reshape(-1, 1)
# print e
print f(e)
