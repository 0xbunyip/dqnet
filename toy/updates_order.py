import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict

a = theano.shared(np.zeros((1, 1), dtype = theano.config.floatX))
b = theano.shared(np.zeros((1, 1), dtype = theano.config.floatX))
c = T.scalar()
updates = OrderedDict()
updates[a] = a + 1
updates[b] = b + a
f = theano.function([c], c, updates = updates)

print f(1)
print a.get_value()
print b.get_value()
print f(2)
print a.get_value()
print b.get_value()
print f(3)
print a.get_value()
print b.get_value()