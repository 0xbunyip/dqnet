import numpy as np
import theano
import theano.tensor as T
import lasagne

batch_size = 32
input_size = 128
network = lasagne.layers.InputLayer((batch_size, input_size))
network = lasagne.layers.DenseLayer(network, num_units = 512
								, nonlinearity = lasagne.nonlinearities.rectify
								, W = lasagne.init.HeUniform('relu')
								, b = None)
								# , b = lasagne.init.Constant(0.1))
network = lasagne.layers.BiasLayer(network, shared_axes=(0, 1))
print lasagne.layers.get_output_shape(network)

params = lasagne.layers.get_all_param_values(network)
for i in xrange(len(params)):
	print i, params[i].shape