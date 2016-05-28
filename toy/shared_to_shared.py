import numpy as np
import theano
import theano.tensor as T

np.random.seed(333)

s = theano.shared(np.arange(12).astype(np.float32))
d = T.cast(theano.shared(np.arange(5).astype(np.float32)), 'int32')
r = s[d]
f = theano.function([], r)

print f()

