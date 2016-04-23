import numpy as np
import theano
import theano.tensor as T
import lasagne

np.random.seed(456)

mbsize = 10
channel = 3
height = 8
width = 8
num_action = 4
gamma = 0.9

input_var = T.tensor4(dtype = 'float32')
action_var = T.col(dtype = 'uint8')
terminal_var = T.col(dtype = 'int8')

network = lasagne.layers.InputLayer(
			(mbsize, channel, height, width), input_var)
network = lasagne.layers.Conv2DLayer(
	network, num_filters = 3, filter_size = (2, 2), stride = (1, 1))
network = lasagne.layers.DenseLayer(
	network, num_units = 10)
network = lasagne.layers.DenseLayer(
	network, num_units = num_action, nonlinearity = None)
network_out = lasagne.layers.get_output(network)
network_indexing = network_out[T.arange(mbsize), action_var.reshape((-1, ))]
network_max = T.max(network_out, axis = 1).reshape((-1, 1))
network_discount = gamma * network_max * (T.ones_like(terminal_var) - terminal_var)

f = theano.function([input_var], network_out)
g = theano.function([input_var, action_var], network_indexing)
h = theano.function([input_var], network_max)
j = theano.function([input_var, terminal_var], network_discount)

inp = np.uint8(np.random.randint(0, 256, (mbsize, channel, height, width)))
print "inp.shape", inp.shape

act = np.uint8(np.random.randint(0, num_action, (mbsize, 1)))
print act.reshape(-1, )

out = f(inp)
print "out", out
print "out.shape", out.shape

out_id = g(inp, act)
print "out_id", out_id
print "out_id.shape", out_id.shape

out_id_correct = out[np.arange(mbsize), act.reshape(-1, )]
print "out_id_correct", out_id_correct
print "out_id_correct.shape", out_id_correct.shape
print "Correct =", np.allclose(out_id, out_id_correct)

out_max = h(inp)
print "out_max", out_max
print "out_max.shape", out_max.shape

term = np.int8(np.random.randint(0, 5, (mbsize, 1)) == 1)
print "term", term
out_discount = j(inp, term)
print "out_discount", out_discount
print "out_discount.shape", out_discount.shape