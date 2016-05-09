import numpy as np
import lasagne
import theano
import theano.tensor as T
from collections import OrderedDict

from theano.printing import debugprint

class Network:
	"""docstring for Network"""

	GRAD_MOMENTUM = 0.95
	LEARNING_RATE = 0.00025
	MAX_ERROR = 0.0
	MIN_SQR_GRAD = 0.01
	SCALE_FACTOR = 255.0
	SQR_GRAD_MOMENTUM = 0.95
	TARGET_NETWORK_UPDATE_FREQUENCY = 10000

	def __init__(self, num_action, mbsize, channel, height, width, discount, up_freq, rng, network_type, init_params = None):
		self.num_action = num_action
		self.height = height
		self.width = width
		self.mbsize = mbsize
		self.channel = channel
		self.discount = discount
		self.up_freq = up_freq
		self.freeze = Network.TARGET_NETWORK_UPDATE_FREQUENCY / up_freq
		self.network_type = network_type
		self.rng = rng
		self.train_count = 0
		lasagne.random.set_rng(rng)
		self.network_description = ''

		if network_type == 'nature':
			self.net = self._build_nature_dqn(num_action, channel, height, width)
			self.tnet = self._build_nature_dqn(num_action, channel, height, width) if self.freeze > 0 else None
		elif network_type == 'nips':
			self.net = self._build_nips_dqn(num_action, channel, height, width)
			self.tnet = self._build_nips_dqn(num_action, channel, height, width) if self.freeze > 0 else None
		elif network_type == 'simple':
			self.net = self._build_simple_network(num_action, channel, height, width)
			self.tnet = self._build_simple_network(num_action, channel, height, width) if self.freeze > 0 else None
		elif network_type == 'bandit':
			self.net = self._build_bandit_network(num_action, channel, height, width)
			self.tnet = self._build_bandit_network(num_action, channel, height, width) if self.freeze > 0 else None
		elif network_type == 'grid':
			self.net = self._build_grid_network(num_action, channel, height, width)
			self.tnet = self._build_grid_network(num_action, channel, height, width) if self.freeze > 0 else None
		elif network_type == 'linear':
			self.net = self._build_linear_network(num_action, channel, height, width)
			self.tnet = self._build_linear_network(num_action, channel, height, width) if self.freeze > 0 else None
		
		if init_params is not None:
			self.set_params(init_params)

		if self.freeze > 0:
			self.clone_target()

		self.shared_single = theano.shared(
			np.zeros((1, channel, height, width), 
			dtype = np.float32))
		self.shared_states = theano.shared(
			np.zeros((mbsize, channel + 1, height, width), 
			dtype = np.float32))
		self.shared_action = theano.shared(
			np.zeros((mbsize, 1), 
			dtype = np.uint8), broadcastable=(False, True))
		self.shared_reward = theano.shared(
			np.zeros((mbsize, 1), 
			dtype = np.float32), broadcastable=(False, True))
		self.shared_terminal = theano.shared(
			np.zeros((mbsize, 1), 
			dtype = np.int8), broadcastable=(False, True))

		self.max_error = np.float32(Network.MAX_ERROR)
		self.train_fn = self._compile_train_function()
		self.evaluate_fn = self._compile_evaluate_function()
		self.validate_fn = self._compile_validate_function()

		self.action_values_fn = self._compile_action_values_function()

		# with open('graph.txt', 'w') as f:
		# 	debugprint(self.validate_fn, print_type = True, file = f)

		print "Finished building network"

	def _build_nature_dqn(self, num_action, channel, height, width):
		self.network_description = 'Nature deep Q-network'
		network = lasagne.layers.InputLayer((None, channel, height, width))

		network = lasagne.layers.Conv2DLayer(network
			, num_filters = 32, filter_size = (8, 8), stride = (4, 4)
			, nonlinearity = lasagne.nonlinearities.rectify
			, W = lasagne.init.HeUniform('relu')
			, b = lasagne.init.Constant(0.1))

		network = lasagne.layers.Conv2DLayer(network
			, num_filters = 64, filter_size = (4, 4), stride = (2, 2)
			, nonlinearity = lasagne.nonlinearities.rectify
			, W = lasagne.init.HeUniform('relu')
			, b = lasagne.init.Constant(0.1))

		network = lasagne.layers.Conv2DLayer(network
			, num_filters = 64, filter_size = (3, 3), stride = (1, 1)
			, nonlinearity = lasagne.nonlinearities.rectify
			, W = lasagne.init.HeUniform('relu')
			, b = lasagne.init.Constant(0.1))

		network = lasagne.layers.DenseLayer(network
			, num_units = 512
			, nonlinearity = lasagne.nonlinearities.rectify
			, W = lasagne.init.HeUniform('relu')
			, b = lasagne.init.Constant(0.1))

		network = lasagne.layers.DenseLayer(network
			, num_units = num_action
			, nonlinearity = None
			, W = lasagne.init.HeUniform()
			, b = lasagne.init.Constant(0.1))
		return network

	def _build_nips_dqn(self, num_action, channel, height, width):
		self.network_description = 'NIPS deep Q-network'
		network = lasagne.layers.InputLayer((None, channel, height, width))

		network = lasagne.layers.Conv2DLayer(network
			, num_filters = 16, filter_size = (8, 8), stride = (4, 4)
			, nonlinearity = lasagne.nonlinearities.rectify
			, W = lasagne.init.HeUniform('relu')
			, b = lasagne.init.Constant(0.1))

		network = lasagne.layers.Conv2DLayer(network
			, num_filters = 32, filter_size = (4, 4), stride = (2, 2)
			, nonlinearity = lasagne.nonlinearities.rectify
			, W = lasagne.init.HeUniform('relu')
			, b = lasagne.init.Constant(0.1))

		network = lasagne.layers.DenseLayer(network
			, num_units = 256
			, nonlinearity = lasagne.nonlinearities.rectify
			, W = lasagne.init.HeUniform('relu')
			, b = lasagne.init.Constant(0.1))

		network = lasagne.layers.DenseLayer(network
			, num_units = num_action
			, nonlinearity = None
			, W = lasagne.init.HeUniform()
			, b = lasagne.init.Constant(0.1))
		return network

	def _build_simple_network(self, num_action, channel, height, width):
		self.network_description = 'Simple network [Conv(4, 2, 2, 1, 1), Dense(256)]'
		network = lasagne.layers.InputLayer((None, channel, height, width))

		network = lasagne.layers.Conv2DLayer(network
			, num_filters = 4, filter_size = (2, 2), stride = (1, 1)
			, nonlinearity = lasagne.nonlinearities.rectify
			, W = lasagne.init.HeUniform('relu')
			, b = lasagne.init.Constant(0.1))

		network = lasagne.layers.DenseLayer(network
			, num_units = 256
			, nonlinearity = lasagne.nonlinearities.rectify
			, W = lasagne.init.HeUniform('relu')
			, b = lasagne.init.Constant(0.1))

		network = lasagne.layers.DenseLayer(network
			, num_units = num_action
			, nonlinearity = None
			, W = lasagne.init.HeUniform()
			, b = lasagne.init.Constant(0.1))
		return network

	def _build_linear_network(self, num_action, channel, height, width):
		self.network_description = 'Linear network'
		network = lasagne.layers.InputLayer((None, channel, height, width))

		network = lasagne.layers.DenseLayer(network
			, num_units = num_action
			, nonlinearity = None
			, W = lasagne.init.HeUniform()
			, b = None)
			# , b = lasagne.init.Constant(0.1))
		return network

	def _build_bandit_network(self, num_action, channel, height, width):
		self.network_description = 'Bandit network [Dense(128), Dense(128)]'
		network = lasagne.layers.InputLayer((None, channel, height, width))

		network = lasagne.layers.DenseLayer(network
			, num_units = 128
			, nonlinearity = lasagne.nonlinearities.rectify
			, W = lasagne.init.HeUniform('relu')
			, b = lasagne.init.Constant(0.1))

		network = lasagne.layers.DenseLayer(network
			, num_units = 128
			, nonlinearity = lasagne.nonlinearities.rectify
			, W = lasagne.init.HeUniform('relu')
			, b = lasagne.init.Constant(0.1))

		network = lasagne.layers.DenseLayer(network
			, num_units = num_action
			, nonlinearity = None
			, W = lasagne.init.HeUniform()
			, b = lasagne.init.Constant(0.1))
		return network

	def _build_grid_network(self, num_action, channel, height, width):
		self.network_description = 'Grid network [Conv(32, 2, 2, 1, 1), Dense(256)]'
		network = lasagne.layers.InputLayer((None, channel, height, width))

		network = lasagne.layers.Conv2DLayer(network
			, num_filters = 32, filter_size = (2, 2), stride = (1, 1)
			, nonlinearity = lasagne.nonlinearities.rectify
			, W = lasagne.init.HeUniform('relu')
			, b = lasagne.init.Constant(0.1))

		network = lasagne.layers.DenseLayer(network
			, num_units = 256
			, nonlinearity = lasagne.nonlinearities.rectify
			, W = lasagne.init.HeUniform('relu')
			, b = lasagne.init.Constant(0.1))

		network = lasagne.layers.DenseLayer(network
			, num_units = num_action
			, nonlinearity = None
			, W = lasagne.init.HeUniform()
			, b = lasagne.init.Constant(0.1))

		return network

	def _compile_train_function(self):
		state = T.tensor4(dtype = theano.config.floatX)
		action = T.col(dtype = 'uint8')
		reward = T.col(dtype = theano.config.floatX)
		terminal = T.col(dtype = 'int8')
		next_state = T.tensor4(dtype = theano.config.floatX)

		current_values_matrix = lasagne.layers.get_output(self.net, state)
		action_mask = T.eq(T.arange(self.num_action).reshape((1, -1)), action.reshape((-1, 1))).astype(theano.config.floatX)
		current_values = T.sum(current_values_matrix * action_mask, axis = 1).reshape((-1, 1))

		if self.tnet is not None:
			target_values_matrix = lasagne.layers.get_output(self.tnet, next_state)
		else:
			target_values_matrix = lasagne.layers.get_output(self.net, next_state)

		bootstrap_values = T.max(target_values_matrix, axis = 1, keepdims = True)
		terminal_floatX = terminal.astype(theano.config.floatX)
		target_values = reward + self.discount * (T.ones_like(terminal_floatX) - terminal_floatX) * bootstrap_values

		if self.tnet is None:
			target_values = theano.gradient.disconnected_grad(target_values)

		error = target_values - current_values
		if self.max_error > 0:
			# From https://github.com/spragunr/deep_q_rl/issues/46
			quadratic_term = T.minimum(abs(error), self.max_error)
			linear_term = abs(error) - quadratic_term
			loss = T.sum(0.5 * quadratic_term ** 2 + linear_term * self.max_error)
		else:
			loss = T.sum(0.5 * error ** 2)

		net_params = lasagne.layers.get_all_params(self.net)
		updates = self._get_rmsprop_updates(loss, net_params, 
			lr = Network.LEARNING_RATE, grad_momentum = Network.GRAD_MOMENTUM, 
			sqr_momentum = Network.SQR_GRAD_MOMENTUM, min_grad = Network.MIN_SQR_GRAD)

		train_givens = {
			state : self.shared_states[:, :-1, :, :] / Network.SCALE_FACTOR, 
			action : self.shared_action, 
			reward : self.shared_reward, 
			terminal : self.shared_terminal, 
			next_state : self.shared_states[:, 1:, :, :] / Network.SCALE_FACTOR, 
		}
		return theano.function([], loss, updates = updates, givens = train_givens)

	def _get_rmsprop_updates(self, loss, params, lr, grad_momentum, sqr_momentum, min_grad):
		# Modified from the Lasagne package:
		# 	https://github.com/Lasagne/Lasagne/blob/master/lasagne/updates.py

		grads = theano.grad(loss, params)
		updates = OrderedDict()

		# Using theano constant to prevent upcasting of float32
		one = T.constant(1)
		for param, grad in zip(params, grads):
			value = param.get_value(borrow=True)
			accu_sqr = theano.shared(np.zeros(value.shape, dtype=value.dtype),
				broadcastable=param.broadcastable)
			accu_sqr_new = sqr_momentum * accu_sqr + (one - sqr_momentum) * grad ** 2

			accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
				broadcastable=param.broadcastable)
			accu_new = grad_momentum * accu + (one - grad_momentum) * grad

			updates[accu] = accu_new
			updates[accu_sqr] = accu_sqr_new
			updates[param] = param - (lr * grad /
				T.sqrt(accu_sqr_new - accu_new ** 2 + min_grad))
		return updates

	def _compile_evaluate_function(self):
		state = T.tensor4(dtype = theano.config.floatX)
		action_values_matrix = lasagne.layers.get_output(self.net, state)
		action_to_take = T.argmax(action_values_matrix, axis = 1)[0]
		return theano.function([], action_to_take
			, givens = {state : self.shared_single / Network.SCALE_FACTOR})

	def _compile_validate_function(self):
		state = T.tensor4(dtype = theano.config.floatX)
		action_values_matrix = lasagne.layers.get_output(self.net, state)
		max_action_values = T.max(action_values_matrix, axis = 1)
		return theano.function([], max_action_values
			, givens = {state : self.shared_states[:, :-1, :, :] / Network.SCALE_FACTOR})

	def _compile_action_values_function(self):
		state = T.tensor4(dtype = theano.config.floatX)
		action_values_matrix = lasagne.layers.get_output(self.net, state)
		return theano.function([], action_values_matrix
			, givens = {state : self.shared_states[:, :-1, :, :] / Network.SCALE_FACTOR})

	def train_one_minibatch(self, states, action, reward, terminal):
		assert self.train_fn is not None
		self.train_count += 1
		if self.freeze > 0 and self.train_count % self.freeze == 0:
			self.clone_target()

		self.shared_states.set_value(states)
		self.shared_action.set_value(action)
		self.shared_reward.set_value(reward)
		self.shared_terminal.set_value(terminal)
		loss = self.train_fn()
		return loss

	def get_action(self, single_state):
		assert self.evaluate_fn is not None
		# self.shared_single.set_value(single_state[np.newaxis, ...] / np.float32(Network.SCALE_FACTOR))
		self.shared_single.set_value(single_state[np.newaxis, ...])
		return self.evaluate_fn()

	def get_max_action_values(self, states):
		assert self.validate_fn is not None
		# self.shared_states.set_value(states / np.float32(Network.SCALE_FACTOR))
		self.shared_states.set_value(states)
		return self.validate_fn()

	def get_action_values(self, states):
		assert self.action_values_fn is not None
		# self.shared_states.set_value(states / np.float32(Network.SCALE_FACTOR))
		self.shared_states.set_value(states)
		return self.action_values_fn()

	def clone_target(self):
		assert self.tnet is not None
		# print "\tCloning network"
		lasagne.layers.set_all_param_values(self.tnet, self.get_params())

	def get_params(self):
		return lasagne.layers.get_all_param_values(self.net)

	def set_params(self, params):
		lasagne.layers.set_all_param_values(self.net, params)
		
		