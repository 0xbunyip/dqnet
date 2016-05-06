import numpy as np
import theano
import theano.tensor as T
from theano.printing import debugprint

a = T.matrix(dtype = theano.config.floatX, name = 'a')
c = T.iscalar(name = 'c')
d = c.astype(theano.config.floatX)
b = a * (T.ones_like(d) - d)
# debugprint(b
f = theano.function([a, c], b)
debugprint(f, print_type = True)
# print theano.pp(f.maker.fgraph.outputs[0])
x = np.array([[5.0, 3.0], [2.5, -1.0]], dtype = np.float32)
y = 0
print x, y
print f(x, y)