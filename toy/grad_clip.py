import numpy as np
import theano
import theano.tensor as T
import lasagne
from collections import OrderedDict
from lasagne.utils import compute_norms
from theano.printing import debugprint

rng = np.random.RandomState(333)
mbsize = 1
input_size = 1

input_var = T.matrix(dtype = theano.config.floatX)
network = lasagne.layers.InputLayer((mbsize, input_size))
network = lasagne.layers.DenseLayer(network
									, num_units = 1
									, nonlinearity = None
									, W = lasagne.init.Constant(1.0)
									, b = lasagne.init.Constant(0.0))
loss = T.sum(lasagne.layers.get_output(network, input_var)) * 1000

print "\nparam_values before update"
param_values = lasagne.layers.get_all_param_values(network)
print param_values

params = lasagne.layers.get_all_params(network)
updates = OrderedDict()
grads = theano.grad(loss, params)
grads = lasagne.updates.total_norm_constraint(grads, 10)

for param, grad in zip(params, grads):
	# debugprint(grad)
	updates[param] = param - 1. * grad

f = theano.function([input_var], [], updates = updates)
# input_values = rng.rand(mbsize, input_size).astype(np.float32)
input_values = np.ones((mbsize, input_size)).astype(np.float32) * 2
_ = f(input_values)

print "\nparam_values after update"
param_values = lasagne.layers.get_all_param_values(network)
print param_values
