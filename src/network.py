import numpy as np
import lasagne
import theano
import theano.tensor as T
import cPickle
import os
from collections import OrderedDict
from util.duel_aggregator import DuelAggregateLayer

from theano.printing import debugprint

class Network:
	"""docstring for Network"""

	CLONE_FREQ = 10000
	GRAD_MOMENTUM = 0.95
	INPUT_SCALE = 255.0
	LEARNING_RATE = 0.00025
	MAX_ERROR = 1.0
	MAX_NORM = 10.0
	MIN_SGRAD = 0.01
	SGRAD_MOMENTUM = 0.95
	
	def __init__(self, num_action, mbsize, channel, height, width, discount
				, up_freq, rng, network_type, algorithm
				, network_file = None, num_ignore = 0):
		self.num_action = num_action
		self.height = height
		self.width = width
		self.mbsize = mbsize
		self.channel = channel
		self.discount = discount
		self.up_freq = up_freq
		self.freeze = Network.CLONE_FREQ / up_freq
		self.max_norm = Network.MAX_NORM
		self.network_type = network_type
		self.algorithm = algorithm
		self.rng = rng
		self.train_count = 0
		self.learning_rate = Network.LEARNING_RATE
		lasagne.random.set_rng(rng)
		self.network_description = ''
		self.adv_net = None
		self.adv_tnet = None

		if self.network_type == 'duel':
			self.net, self.adv_net = self._build_duel_dqn(num_action, channel
															, height, width)
			self.tnet, self.adv_tnet = self._build_duel_dqn(num_action, channel
															, height, width) \
												if self.freeze > 0 else None
		elif self.network_type == 'double':
			self.net = self._build_double_dqn(num_action, channel, height, width)
			self.tnet = self._build_double_dqn(num_action, channel, height, width) \
												if self.freeze > 0 else None
		elif self.network_type == 'nature':
			self.net = self._build_nature_dqn(num_action, channel, height, width)
			self.tnet = self._build_nature_dqn(num_action, channel, height, width) \
												if self.freeze > 0 else None
		elif self.network_type == 'nips':
			self.net = self._build_nips_dqn(num_action, channel, height, width)
			self.tnet = self._build_nips_dqn(num_action, channel, height, width) \
												if self.freeze > 0 else None
		elif self.network_type == 'simple':
			self.net = self._build_simple_network(num_action, channel, height, width)
			self.tnet = self._build_simple_network(num_action, channel, height, width) \
												if self.freeze > 0 else None
		elif self.network_type == 'bandit':
			self.net = self._build_bandit_network(num_action, channel, height, width)
			self.tnet = self._build_bandit_network(num_action, channel, height, width) \
												if self.freeze > 0 else None
		elif self.network_type == 'grid':
			self.net = self._build_grid_network(num_action, channel, height, width)
			self.tnet = self._build_grid_network(num_action, channel, height, width) \
												if self.freeze > 0 else None
		elif self.network_type == 'linear':
			self.net = self._build_linear_network(num_action, channel, height, width)
			self.tnet = self._build_linear_network(num_action, channel, height, width) \
												if self.freeze > 0 else None
		
		self.transfer_desc = ""
		if network_file is not None:
			self._init_params(network_file, num_ignore)

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

	def get_info(self):
		info = self.network_description + '\n'
		info += self.transfer_desc + '\n'
		info += self.algorithm + '\n\n'
		return info

	def _init_params(self, network_file, num_ignore):
		print "Loading network params from file"
		_, file_ext = os.path.splitext(network_file)
		params = []
		default_params = self.get_params()

		# Legacy: read from cPickle file
		if file_ext == '.pkl':
			with open(network_file, 'rb') as f:
				network_type = cPickle.load(f)
				params = cPickle.load(f)
			# Ignore last 'num_ignore' layers' params (both weights and biases)
			for _ in xrange(num_ignore * 2):
				params.pop()
		elif file_ext == '.npz':
			npz = np.load(network_file)
			params = []
			num_layers = len(default_params) // 2 - num_ignore
			# Load the first 'num_layers' layers and discard the rest
			for i in xrange(num_layers):
				params.append(npz['w' + str(i)]) # Weights
				params.append(npz['b' + str(i)]) # Biases

		self.transfer_desc = "Load network from " + network_file
		self.transfer_desc += "\nIgnore last %d layer(s)" % (num_ignore)

		# Use default values of params for discarded layers
		for i in reversed(range(num_ignore)):
			params.append(default_params[-i * 2 - 2]) # Weights
			params.append(default_params[-i * 2 - 1]) # Biases

		self.set_params(params)

	def dump(self, file_name):
		params = self.get_params()
		arrays = {}
		for i, p in enumerate(params):
			name = 'w' if i % 2 == 0 else 'b'
			name = name + str(i // 2)
			arrays[name] = p
		np.savez_compressed(file_name, **arrays)

	def _get_action_var(self, network, state):
		return T.argmax(lasagne.layers.get_output(network, state)
						, axis = 1, keepdims = True)

	def _compile_train_function(self):
		state = T.tensor4(dtype = theano.config.floatX)
		action = T.col(dtype = 'uint8')
		reward = T.col(dtype = theano.config.floatX)
		terminal = T.col(dtype = 'int8')
		next_state = T.tensor4(dtype = theano.config.floatX)

		current_values_matrix = lasagne.layers.get_output(self.net, state)
		action_mask = T.eq(T.arange(self.num_action).reshape((1, -1))
						, action.reshape((-1, 1))).astype(theano.config.floatX)
		current_values = T.sum(current_values_matrix * action_mask
								, axis = 1).reshape((-1, 1))

		if self.algorithm == 'q_learning':
			if self.tnet is not None:
				target_values = lasagne.layers.get_output(self.tnet, next_state)
			else:
				target_values = lasagne.layers.get_output(self.net, next_state)
			bootstrap_values = T.max(target_values, axis = 1, keepdims = True)

		elif self.algorithm == 'double_q_learning':
			if self.network_type == 'duel':
				# Get argmax actions from advantage values
				select_actions = self._get_action_var(self.adv_net, next_state)
			else:
				# Get argmax actions from Q values
				select_actions = self._get_action_var(self.net, next_state)
			select_mask = T.eq(T.arange(self.num_action).reshape((1, -1))
								, select_actions.astype(theano.config.floatX))

			if self.tnet is not None:
				# Evaluate argmax actions on target network
				eval_values = lasagne.layers.get_output(self.tnet, next_state)
			else:
				# Evaluate argmax actions on online network
				# (the same as q_learning but slower)
				eval_values = lasagne.layers.get_output(self.net, next_state)

			bootstrap_values = T.sum(eval_values * select_mask
									, axis = 1, keepdims = True)

		terminal_floatX = terminal.astype(theano.config.floatX)
		target_values = reward + self.discount * \
			(T.ones_like(terminal_floatX) - terminal_floatX) * bootstrap_values

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
		updates = self._get_rmsprop_updates(loss, net_params
			, lr = Network.LEARNING_RATE, grad_momentum = Network.GRAD_MOMENTUM
			, sqr_momentum = Network.SGRAD_MOMENTUM
			, min_grad = Network.MIN_SGRAD)

		train_givens = {
			state : self.shared_states[:, :-1, :, :] / Network.INPUT_SCALE,
			action : self.shared_action,
			reward : self.shared_reward,
			terminal : self.shared_terminal,
			next_state : self.shared_states[:, 1:, :, :] / Network.INPUT_SCALE,
		}
		return theano.function([], loss, updates = updates, givens = train_givens)

	def _grad_clip_norm(tensor_vars, max_norm):
		norm = T.sqrt(sum(T.sum(tensor ** 2) for tensor in tensor_vars))
		dtype = np.dtype(theano.config.floatX).type
		return max_norm / T.max(dtype(max_norm), norm)

	def _get_rmsprop_updates(self, loss, params, lr, grad_momentum
							, sqr_momentum, min_grad):
		# Modified from the Lasagne package:
		# 	https://github.com/Lasagne/Lasagne/blob/master/lasagne/updates.py

		grads = theano.grad(loss, params)
		scale_factor = 1.0
		if self.max_norm > 0:
			scale_factor = self._grad_clip_norm(grads, self.max_norm)
		updates = OrderedDict()

		# Using theano constant to prevent upcasting of float32
		one = T.constant(1)
		for param, grad in zip(params, grads):
			value = param.get_value(borrow=True)
			accu_sqr = theano.shared(np.zeros(value.shape, dtype=value.dtype),
				broadcastable=param.broadcastable)
			accu_sqr_new = sqr_momentum * accu_sqr + \
							(one - sqr_momentum) * grad ** 2

			accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
				broadcastable=param.broadcastable)
			accu_new = grad_momentum * accu + (one - grad_momentum) * grad

			updates[accu] = accu_new
			updates[accu_sqr] = accu_sqr_new
			updates[param] = param - (lr * grad * scale_factor /
				T.sqrt(accu_sqr_new - accu_new ** 2 + min_grad))
		return updates

	def _compile_evaluate_function(self):
		state = T.tensor4(dtype = theano.config.floatX)
		if self.network_type == 'duel':
			action = self._get_action_var(self.adv_net, state)
		else:
			action = self._get_action_var(self.net, state)
		return theano.function([], action
			, givens = {state : self.shared_single / Network.INPUT_SCALE})

	def _compile_validate_function(self):
		state = T.tensor4(dtype = theano.config.floatX)
		action_values_matrix = lasagne.layers.get_output(self.net, state)
		max_action_values = T.max(action_values_matrix, axis = 1)
		return theano.function([], max_action_values
			, givens = {state : self.shared_states[:, :-1, :, :] /\
						 Network.INPUT_SCALE})

	def _compile_action_values_function(self):
		state = T.tensor4(dtype = theano.config.floatX)
		action_values_matrix = lasagne.layers.get_output(self.net, state)
		return theano.function([], action_values_matrix
			, givens = {state : self.shared_states[:, :-1, :, :] /\
						 Network.INPUT_SCALE})

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
		self.shared_single.set_value(single_state[np.newaxis, ...])
		return self.evaluate_fn()

	def get_max_action_values(self, states):
		assert self.validate_fn is not None
		self.shared_states.set_value(states)
		return self.validate_fn()

	def get_action_values(self, states):
		assert self.action_values_fn is not None
		self.shared_states.set_value(states)
		return self.action_values_fn()

	def clone_target(self):
		assert self.tnet is not None
		lasagne.layers.set_all_param_values(self.tnet, self.get_params())

	def get_params(self):
		return lasagne.layers.get_all_param_values(self.net)

	def set_params(self, params):
		lasagne.layers.set_all_param_values(self.net, params)

	def _build_duel_dqn(self, num_action, channel, height, width):
		self.network_description = "Dueling network architecture"
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

		value_stream = lasagne.layers.DenseLayer(network
			, num_units = 512
			, nonlinearity = lasagne.nonlinearities.rectify
			, W = lasagne.init.HeUniform('relu')
			, b = lasagne.init.Constant(0.1))

		value_stream = lasagne.layers.DenseLayer(value_stream
			, num_units = 1
			, nonlinearity = None
			, W = lasagne.init.HeUniform()
			, b = lasagne.init.Constant(0.1))

		advantage_stream = lasagne.layers.DenseLayer(network
			, num_units = 512
			, nonlinearity = lasagne.nonlinearities.rectify
			, W = lasagne.init.HeUniform('relu')
			, b = lasagne.init.Constant(0.1))

		advantage_stream = lasagne.layers.DenseLayer(advantage_stream
			, num_units = num_action
			, nonlinearity = None
			, W = lasagne.init.HeUniform()
			, b = lasagne.init.Constant(0.1))

		network = DuelAggregateLayer([value_stream, advantage_stream])

		return network, advantage_stream

	def _build_double_dqn(self, num_action, channel, height, width):
		self.network_description = "Double deep Q-network (with shared bias)"
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
			, b = None)

		network = lasagne.layers.BiasLayer(network
			, b = lasagne.init.Constant(0.1)
			, shared_axes=(0, 1))
		return network

	def _build_nature_dqn(self, num_action, channel, height, width):
		self.network_description = "Nature deep Q-network"
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
		self.network_description = "NIPS deep Q-network"
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
		self.network_description = "Simple network [Conv(4, 2, 2, 1, 1)"\
									", Dense(256)]"
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
		self.network_description = "Linear network"
		network = lasagne.layers.InputLayer((None, channel, height, width))

		network = lasagne.layers.DenseLayer(network
			, num_units = num_action
			, nonlinearity = None
			, W = lasagne.init.HeUniform()
			, b = None)
		return network

	def _build_bandit_network(self, num_action, channel, height, width):
		self.network_description = "Bandit network [Dense(128), Dense(128)]"
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
		self.network_description = "Grid network [Conv(32, 2, 2, 1, 1)"\
									", Dense(256)]"
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
