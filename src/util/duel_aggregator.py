import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import MergeLayer

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
