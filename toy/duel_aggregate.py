import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import MergeLayer
from theano.printing import debugprint

class DuelAggregateLayer(MergeLayer):
	def __init__(self, incomings, **kwargs):
		super(DuelAggregateLayer, self).__init__(incomings, **kwargs)

	def get_output_shape_for(self, input_shapes):
		output_shape = tuple(max(dim_sizes) for dim_sizes in zip(*input_shapes))
		return output_shape

	def get_output_for(self, inputs, **kwargs):
		value_stream = T.patternbroadcast(inputs[0], (False, True))
		advantage_stream = inputs[1]
		advantage_mean = T.mean(advantage_stream, axis = 1, keepdims = True)
		advantage_stream = advantage_stream - advantage_mean
		return T.add(value_stream, advantage_stream)

rng = np.random.RandomState(333)

batch_size = 32
input_size = 512
value_size = 1
advantage_size = 18

input_symbol = T.matrix(dtype = theano.config.floatX)

network = lasagne.layers.InputLayer((batch_size, input_size))
value_stream = lasagne.layers.DenseLayer(network, num_units = value_size
								, nonlinearity = None
								, W = lasagne.init.HeUniform()
								, b = lasagne.init.Constant(0.1))
advantage_stream = lasagne.layers.DenseLayer(network, num_units = advantage_size
								, nonlinearity = None
								, W = lasagne.init.HeUniform()
								, b = lasagne.init.Constant(0.1))
duel = DuelAggregateLayer([value_stream, advantage_stream])

print "\nvalue_stream"
print lasagne.layers.get_output_shape(value_stream)
debugprint(lasagne.layers.get_output(value_stream), print_type = True)

print "\nadvantage_stream"
print lasagne.layers.get_output_shape(advantage_stream)
debugprint(lasagne.layers.get_output(advantage_stream), print_type = True)

print "\nduel"
print lasagne.layers.get_output_shape(duel)
debugprint(lasagne.layers.get_output(duel), print_type = True)

f = theano.function([input_symbol], lasagne.layers.get_output(duel, input_symbol))
g = theano.function([input_symbol], lasagne.layers.get_output(value_stream, input_symbol))
h = theano.function([input_symbol], lasagne.layers.get_output(advantage_stream, input_symbol))

print "\nout"
input_value = rng.rand(batch_size, input_size).astype(np.float32)
out = f(input_value)
print out.shape

print "\ncheck"
value_out = g(input_value)
advantage_out = h(input_value)
sum_out = value_out + advantage_out - np.mean(advantage_out, axis = 1, keepdims = True)
print value_out.shape
print advantage_out.shape
np.testing.assert_array_almost_equal(sum_out, out)
print "pass"

print "\nparams"
params = lasagne.layers.get_all_param_values(duel)
print len(params)
for p in params:
	print p.shape
