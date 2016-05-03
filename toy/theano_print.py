import numpy as np
import theano
import theano.tensor as T

a = T.matrix(dtype = theano.config.floatX, name = 'a')
b = T.matrix(dtype = theano.config.floatX, name = 'b')
c = T.matrix(dtype = theano.config.floatX, name = 'c')
d = T.exp((a + b) * c)
d = T.log(d)
f = theano.function([a, b, c], d)

x = np.random.randint(0, 10, size = (3, 3)).astype(np.float32)
y = np.random.randint(0, 10, size = (3, 3)).astype(np.float32)
z = np.random.randint(0, 10, size = (3, 3)).astype(np.float32)

print x, y, z
print f(x, y, z)
print "\nDebug print f =\n", theano.printing.debugprint(f)
print "\nPretty print f.maker.fgraph.outputs[0] =\n", theano.pp(f.maker.fgraph.outputs[0])
print "\nPretty print f.maker.fgraph.toposort() =\n", f.maker.fgraph.toposort()

