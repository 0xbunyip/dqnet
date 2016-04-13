import numpy as np
import theano
import theano.tensor as T
import lasagne

np.random.seed(456)

mbsize = 3
shared_terminal = theano.shared(np.zeros((mbsize, 1), dtype = np.int8), broadcastable=(False, True))
terminal = T.col(dtype = 'int8')
out = terminal + 1
givens = {terminal : shared_terminal}
f = theano.function([], out, givens = givens)
print f()

tmp = np.zeros((mbsize, 1), dtype = np.int8)
g = theano.function([terminal], out)
print g(tmp)
		