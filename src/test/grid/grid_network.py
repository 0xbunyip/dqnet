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
		self.dummy_state = None
		self.shared_state = None
		self.shared_action = None
		self.shared_reward = None
		self.shared_terminal = None
		self.shared_next_state = None

	def _build_dqn(self, input_var, num_action, mbsize, channel, height, width):
		network = lasagne.layers.InputLayer(
			(mbsize, channel, height, width), input_var)
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
			(mbsize, channel, height, width), input_var)
		network = lasagne.layers.Conv2DLayer(
			network, num_filters = 32, filter_size = (2, 2), stride = (1, 1))
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

		####
		self.loss_fn = theano.function([], loss, givens = train_givens)
		self.valmat_fn = theano.function([], current_values_matrix, givens = {state : self.shared_state})
		####
		
		evaluate_givens = {state : self.shared_state}
		self.dummy_state = np.zeros(
			(self.mbsize, self.channel, self.height, self.width), 
			dtype = np.float32)
		self.evaluate_fn = theano.function([], current_values_matrix, givens = evaluate_givens)
		print "Finished building train and evaluate function"

	def train_one_minibatch(self, tnet, state, action, reward, terminal, next_state):
		assert self.train_fn != None
		self.shared_state.set_value(state / np.float32(Network.SCALE_FACTOR))
		self.shared_action.set_value(action)
		self.shared_reward.set_value(reward)
		self.shared_terminal.set_value(terminal)
		self.shared_next_state.set_value(next_state / np.float32(Network.SCALE_FACTOR))

		# valmat_before = self.valmat_fn()
		loss = self.train_fn()
		# valmat_after = self.valmat_fn()

		# print "Current state =\n", state
		# print "Action =", action
		# print "Reward =", reward
		# print "Before =\n", valmat_before
		# print "After =\n", valmat_after
		# raw_input()

		# loss_after = self.loss_fn()
		# print "Loss before = %.3f, loss after = %.3f" % (loss, loss_after)
		# raw_input()
		return loss

	def get_action(self, state, print_Q = False):
		assert self.evaluate_fn != None
		self.dummy_state[0, ...] = state / np.float32(Network.SCALE_FACTOR)
		self.shared_state.set_value(self.dummy_state)
		action_values_matrix = self.evaluate_fn()

		if print_Q:
			print "State =\n", state
			print "Action values matrix =\n", action_values_matrix[0, :]
			print "Action chosen =", np.argmax(action_values_matrix[0, :])
			raw_input()
		return np.argmax(action_values_matrix[0, :])

	def get_params(self):
		return lasagne.layers.get_all_param_values(self.net)

	def set_params(self, params):
		lasagne.layers.set_all_param_values(self.net, params)
		
		