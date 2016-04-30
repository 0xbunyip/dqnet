import numpy as np
import lasagne
import theano
import theano.tensor as T

class Network:
	"""docstring for Network"""

	LEARNING_RATE = 0.0025
	SCALE_FACTOR = 255.0

	def __init__(self, num_action, mbsize, channel, height, width, discount, rng):
		self.num_action = num_action
		self.height = height
		self.width = width
		self.mbsize = mbsize
		self.channel = channel
		self.discount = discount
		self.rng = rng
		lasagne.random.set_rng(rng)
		self.input_var = T.tensor4()
		# self.net = self._build_dqn(self.input_var, num_action, mbsize, channel, height, width)
		# self.net = self._build_simple_network(self.input_var, num_action, mbsize, channel, height, width)
		# self.net = self._build_bandit_network(self.input_var, num_action, mbsize, channel, height, width)
		self.net = self._build_grid_network(self.input_var, num_action, mbsize, channel, height, width)
		self.max_delta = Network.MAX_DELTA
		self.train_fn = None
		self.evaluate_fn = None
		self.validate_fn = None
		self.shared_single = None
		self.shared_states = None
		self.shared_action = None
		self.shared_reward = None
		self.shared_terminal = None
		self.shared_next_state = None

	def _build_dqn(self, input_var, num_action, mbsize, channel, height, width):
		network = lasagne.layers.InputLayer(
			(None, channel, height, width), input_var)
		network = lasagne.layers.Conv2DLayer(
			network, num_filters = 32, filter_size = (8, 8), stride = (4, 4))
		network = lasagne.layers.Conv2DLayer(
			network, num_filters = 64, filter_size = (4, 4), stride = (2, 2))
		network = lasagne.layers.Conv2DLayer(
			network, num_filters = 64, filter_size = (3, 3), stride = (1, 1))
		network = lasagne.layers.DenseLayer(
			network, num_units = 512)
		network = lasagne.layers.DenseLayer(
			network, num_units = num_action, nonlinearity = None)
		return network

	def _build_simple_network(self, input_var, num_action, mbsize, channel, height, width):
		network = lasagne.layers.InputLayer(
			(None, channel, height, width), input_var)
		network = lasagne.layers.Conv2DLayer(
			network, num_filters = 3, filter_size = (2, 2), stride = (1, 1))
		network = lasagne.layers.DenseLayer(
			network, num_units = 50)
		network = lasagne.layers.DenseLayer(
			network, num_units = num_action, nonlinearity = None)
		return network

	def _build_bandit_network(self, input_var, num_action, mbsize, channel, height, width):
		network = lasagne.layers.InputLayer(
			(None, channel, height, width), input_var)
		network = lasagne.layers.DenseLayer(
			network, num_units = 256)
		network = lasagne.layers.DenseLayer(
			network, num_units = num_action, nonlinearity = None)
		return network

	def _build_grid_network(self, input_var, num_action, mbsize, channel, height, width):
		network = lasagne.layers.InputLayer(
			(None, channel, height, width), input_var)
		network = lasagne.layers.Conv2DLayer(
			network, num_filters = 32, filter_size = (2, 2), stride = (1, 1))
		network = lasagne.layers.DenseLayer(
			network, num_units = 512)
		network = lasagne.layers.DenseLayer(
			network, num_units = num_action, nonlinearity = None)
		return network

	def compile_train_function(self, tnet):
		self.shared_single = theano.shared(
			np.zeros((1, self.channel, self.height, self.width), 
			dtype = np.float32))
		self.shared_states = theano.shared(
			np.zeros((self.mbsize, self.channel + 1, self.height, self.width), 
			dtype = np.float32))
		self.shared_action = theano.shared(
			np.zeros((self.mbsize, 1), 
			dtype = np.uint8), broadcastable=(False, True))
		self.shared_reward = theano.shared(
			np.zeros((self.mbsize, 1), 
			dtype = np.int32), broadcastable=(False, True))
		self.shared_terminal = theano.shared(
			np.zeros((self.mbsize, 1), 
			dtype = np.int8), broadcastable=(False, True))

		state = T.tensor4(dtype = 'float32')
		action = T.col(dtype = 'uint8')
		reward = T.col(dtype = 'int32')
		terminal = T.col(dtype = 'int8')
		next_state = T.tensor4(dtype = 'float32')

		self.train_fn = self._compile_train_function(tnet, state, action, reward, terminal, next_state)
		self.evaluate_fn = self._compile_evaluate_function(state)
		self.validate_fn = self._compile_validate_function(state)		

		# theano.printing.debugprint(self.train_fn)
		print "Finished building train and evaluate function"

	def _compile_train_function(self, tnet, state, action, reward, terminal, next_state):
		train_givens = {
			state : self.shared_states[:, :-1, :, :], 
			action : self.shared_action, 
			reward : self.shared_reward, 
			terminal : self.shared_terminal, 
			next_state : self.shared_states[:, 1:, :, :], 
		}
		current_values_matrix = lasagne.layers.get_output(self.net, state)
		action_mask = T.eq(T.arange(self.num_action).reshape((1, -1)), action.reshape((-1, 1))).astype(theano.config.floatX)
		current_values = T.sum(current_values_matrix * action_mask, axis = 1).reshape((-1, 1))
		# current_values = current_values_matrix[T.arange(self.mbsize), action.reshape((-1, ))].reshape((-1, 1))

		target_values_matrix = lasagne.layers.get_output(tnet.net, next_state)
		bootstrap_values = T.max(target_values_matrix, axis = 1, keepdims = True)
		terminal_floatX = terminal.astype(theano.config.floatX)
		target_values = reward + self.discount * (T.ones_like(terminal_floatX) - terminal_floatX) * bootstrap_values		

		error = target_values - current_values
		if self.max_delta > 0:
			quadratic_term = T.minimum(abs(error), self.max_delta)
			linear_term = abs(error) - quadratic_term
			loss = T.sum(0.5 * quadratic_term ** 2 + linear_term * self.max_delta)
		else:
			loss = T.sum(0.5 * error ** 2)

		net_params = lasagne.layers.get_all_params(self.net)
		updates = lasagne.updates.rmsprop(loss, net_params, learning_rate = Network.LEARNING_RATE)

		##########
		# print(theano.pp(current_values_matrix))
		# theano.printing.debugprint(terminal_floatX)

		return theano.function([], loss, updates = updates, givens = train_givens)

	def _compile_evaluate_function(self, state):
		action_values_matrix = lasagne.layers.get_output(self.net, state)
		action_to_take = T.argmax(action_values_matrix, axis = 1)[0]
		return theano.function([], action_to_take, givens = {state : self.shared_single})

	def _compile_validate_function(self, state):
		action_values_matrix = lasagne.layers.get_output(self.net, state)
		max_action_values = T.max(action_values_matrix, axis = 1)
		return theano.function([], max_action_values, \
			givens = {state : self.shared_states[:, :-1, :, :]})

	def train_one_minibatch(self, tnet, states, action, reward, terminal):
		assert self.train_fn is not None
		self.shared_states.set_value(states / np.float32(Network.SCALE_FACTOR))
		self.shared_action.set_value(action)
		self.shared_reward.set_value(reward)
		self.shared_terminal.set_value(terminal)
		loss = self.train_fn()
		return loss

	def get_action(self, single_state):
		assert self.evaluate_fn is not None
		self.shared_single.set_value(single_state[np.newaxis, ...] / np.float32(Network.SCALE_FACTOR))
		return self.evaluate_fn()

	def get_max_action_values(self, states):
		assert self.validate_fn is not None
		self.shared_states.set_value(states / np.float32(Network.SCALE_FACTOR))
		return self.validate_fn()

	def get_params(self):
		return lasagne.layers.get_all_param_values(self.net)

	def set_params(self, params):
		lasagne.layers.set_all_param_values(self.net, params)
		
		