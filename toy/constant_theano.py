import theano
from theano.ifelse import ifelse
a = theano.tensor.vector()
f = theano.function([a], ifelse(1, a + 1, a-1))
print f([3])
theano.printing.debugprint(f)