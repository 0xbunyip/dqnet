import numpy as np
import theano
import theano.tensor as T
from theano.printing import debugprint

w = theano.shared(np.float32(10.0), name = 'w')

a = T.scalar(dtype = theano.config.floatX, name = 'a')
b = T.scalar(dtype = theano.config.floatX, name = 'b')
c = a + b
f = theano.function([a], c, givens = {b : w / 2})

x = T.scalar(dtype = theano.config.floatX, name = 'x')
y = T.scalar(dtype = theano.config.floatX, name = 'y')
y = y / 2
z = x + y
g = theano.function([x], z, givens = {y : w})

print "f(2) =", f(2)
print "w.get_value() =", w.get_value()
print "f.graph ="
debugprint(f, print_type = True)
print "\n"

print "g(2) =", g(2)
print "w.get_value() =", w.get_value()
print "g.graph ="
debugprint(g, print_type = True)
print "\n"
