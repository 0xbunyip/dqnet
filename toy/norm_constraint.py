import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.utils import compute_norms
from theano.printing import debugprint

param = theano.shared(np.random.randn(3, 5).astype(theano.config.floatX))
print "\nparam"
print param.get_value()

update = param + 100
print "\nupdate"
debugprint(update, print_type = True)

update = lasagne.updates.norm_constraint(update, 10)

print "\nnorm_constraint"
debugprint(update, print_type = True)

func = theano.function([], [], updates=[(param, update)])

# Apply constrained update
_ = func()

norms = compute_norms(param.get_value())

print "\nparam"
param_value = param.get_value()
# print compute_norms(param_value).shape
print param_value
print np.isclose(np.max(norms), 10)
