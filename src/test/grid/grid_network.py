import numpy as np
import lasagne
import theano
import theano.tensor as T

class Network:
	"""docstring for Network"""

	LEARNING_RATE = 0.0025
	SCALE_FACTOR = 15.0

	def __init__(self, num_action, mbsize, channel, height, width, discount):
		self.num_action = num_action
		self.height = height
		self.width = width
		self.mbsize = mbsize
		self.channel = channel
		self.discount = discount
		self.input_var = T.tensor4()
		#self.net = self._build_dqn(self.input_var, num_action, mbsize, channel, height, width)
		self.net = self._build_simple_network(self.input_var, num_action, mbsize, channel, height, width)
		self.train_fn = None
		self.evaluate_fn = None
		self.validate_fn = None
		self.dummy_state = None
		self.shared_state = None
		self.shared_action = None
		self.shared_reward = None
		self.shared_terminal = None
		self.shared_next_state = None

	def _build_simple_network(self, input_var, num_action, mbsize, channel, height, width):
		network = lasagne.layers.InputLayer(
			(mbsize, channel, height, width), input_var)
		network = lasagne.layers.Conv2DLayer(
			network, num_filters = 32, filter_size = (2, 2), stride = (1, 1))
		network = lasagne.layers.Conv2DLayer(
			network, num_filters = 64, filter_size = (2, 2), stride = (1, 1))
		network = lasagne.layers.DenseLayer(
			network, num_units = 512)
		network = lasagne.layers.DenseLayer(
			network, num_units = num_action, nonlinearity = None)
		return network

	def compile_train_function(self, tnet):
		self.shared_state = theano.shared(
			np.zeros((self.mbsize, self.channel, self.height, self.width), 
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
		self.shared_next_state = theano.shared(
			np.zeros((self.mbsize, self.channel, self.height, self.width), 
			dtype = np.float32))

		state = T.tensor4(dtype = 'float32')
		action = T.col(dtype = 'uint8')
		reward = T.col(dtype = 'int32')
		terminal = T.col(dtype = 'int8')
		next_state = T.tensor4(dtype = 'float32')

		train_givens = {
			state : self.shared_state, 
			action : self.shared_action, 
			reward : self.shared_reward, 
			terminal : self.shared_terminal, 
			next_state : self.shared_next_state, 
		}
		
		current_values_matrix = lasagne.layers.get_output(self.net, state)
		current_values = current_values_matrix[T.arange(self.mbsize), action.reshape((-1, ))].reshape((-1, 1))

		target_values_matrix = lasagne.layers.get_output(tnet.net, next_state)
		bootstrap_values = T.max(target_values_matrix, axis = 1).reshape((-1, 1))
		target_values = reward + self.discount * (T.ones_like(terminal) - terminal) * bootstrap_values		

		error = target_values - current_values
		loss = T.mean(error ** 2)
		net_params = lasagne.layers.get_all_params(self.net)
		updates = lasagne.updates.rmsprop(loss, net_params, learning_rate = Network.LEARNING_RATE)
		self.train_fn = theano.function([], loss, updates = updates, givens = train_givens)
		
		evaluate_givens = {state : self.shared_state}
		self.dummy_state = np.zeros(
			(self.mbsize, self.channel, self.height, self.width), 
			dtype = np.float32)
		action_to_take = T.argmax(current_values_matrix, axis = 1)
		self.evaluate_fn = theano.function([], action_to_take, givens = evaluate_givens)

		validate_givens = {state : self.shared_state}
		max_action_values = T.max(current_values_matrix, axis = 1)
		self.validate_fn = theano.function([], max_action_values, givens = validate_givens)
		print "Finished building train and evaluate function"

	def train_one_minibatch(self, tnet, state, action, reward, terminal, next_state):
		assert self.train_fn is not None
		self.shared_state.set_value(state / np.float32(Network.SCALE_FACTOR))
		self.shared_action.set_value(action)
		self.shared_reward.set_value(reward)
		self.shared_terminal.set_value(terminal)
		self.shared_next_state.set_value(next_state / np.float32(Network.SCALE_FACTOR))
		loss = self.train_fn()
		return loss

	def get_action(self, state):
		assert self.evaluate_fn is not None
		self.dummy_state[0, ...] = state / np.float32(Network.SCALE_FACTOR)
		self.shared_state.set_value(self.dummy_state)
		return self.evaluate_fn()[0]

	def get_max_action_values(self, states):
		assert self.validate_fn is not None
		self.shared_state.set_value(states / np.float32(Network.SCALE_FACTOR))
		return self.validate_fn()

	def get_params(self):
		return lasagne.layers.get_all_param_values(self.net)

	def set_params(self, params):
		lasagne.layers.set_all_param_values(self.net, params)