import numpy as np
import unittest
import os.path, sys
import theano
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir))
from network import Network

"""
Author: Nathan Sprague
"""
class ChainMDP(object):
	"""Simple markov chain style MDP.  Three "rooms" and one absorbing
	state. States are encoded for the q_network as arrays with
	indicator entries. E.g. [1, 0, 0, 0] encodes state 0, and [0, 1,
	0, 0] encodes state 1.  The absorbing state is [0, 0, 0, 1]

	Action 0 moves the agent left, departing the maze if it is in state 0.
	Action 1 moves the agent to the right, departing the maze if it is in
	state 2.

	The agent receives a reward of .7 for departing the chain on the left, and
	a reward of 1.0 for departing the chain on the right.

	Assuming deterministic actions and a discount rate of .5, the
	correct Q-values are:

	.7|.25,  .35|.5, .25|1.0,  0|0
	"""

	def __init__(self, success_prob=1.0):
		self.num_actions = 2
		self.num_states = 4
		self.success_prob = success_prob

		self.actions = [np.array([[0]], dtype='uint8'),
		np.array([[1]], dtype='uint8')]

		self.reward_zero = np.array([[0]], dtype=theano.config.floatX)
		self.reward_left = np.array([[.7]], dtype=theano.config.floatX)
		self.reward_right = np.array([[1.0]], dtype=theano.config.floatX)

		self.states = []
		for i in range(self.num_states):
			self.states.append(np.zeros((1, 1, 1, self.num_states),
				dtype=theano.config.floatX))
			self.states[-1][0, 0, 0, i] = 1

	def act(self, state, action_index):

		"""
		action 0 is left, 1 is right.
		"""
		state_index =  np.nonzero(state[0, 0, 0, :])[0][0]

		next_index = state_index
		if np.random.random() < self.success_prob:
			next_index = state_index + action_index * 2 - 1

		# Exit left
		if next_index == -1:
			return self.reward_left, self.states[-1], np.array([[True]])

		# Exit right
		if next_index == self.num_states - 1:
			return self.reward_right, self.states[-1], np.array([[True]])

		if np.random.random() < self.success_prob:
			return (self.reward_zero,
				self.states[state_index + action_index * 2 - 1],
				np.array([[False]]))
		else:
			return (self.reward_zero, self.states[state_index],
				np.array([[False]]))

class TestNetwork(unittest.TestCase):
	def setUp(self):
		self.rng = np.random.RandomState(123)
		self.mdp = ChainMDP()
		Network.SCALE_FACTOR = 1.0
		Network.LEARNING_RATE = 0.025
		Network.MAX_ERROR = 0.0
		Network.MIN_SQR_GRAD = 0.01
		Network.GRAD_MOMENTUM = 0.95
		Network.SQR_GRAD_MOMENTUM = 0.95
		Network.TARGET_NETWORK_UPDATE_FREQUENCY = 1

	def get_q_table(self, net):
		mdp = self.mdp
		states = np.zeros((mdp.num_states, 2, 1, mdp.num_states)).astype(theano.config.floatX)
		for i in range(mdp.num_states):
			states[i, 0, ...] = mdp.states[i]
		return net.get_action_values(states)

	def train(self, net, num_step):
		mdp = self.mdp
		for _ in xrange(num_step):
			cstate = mdp.states[self.rng.randint(0, mdp.num_states - 1)]
			action_index = self.rng.randint(0, mdp.num_actions)
			reward, nstate, terminal = mdp.act(cstate, action_index)

			states = np.zeros((1, 2, 1, mdp.num_states)).astype(theano.config.floatX)
			states[:, 0, :, :] = cstate
			states[:, 1, :, :] = nstate
			net.train_one_minibatch(states, mdp.actions[action_index], reward, terminal)

	def test_convergence_no_freeze(self):
		Network.TARGET_NETWORK_UPDATE_FREQUENCY = 0
		net = Network(self.mdp.num_actions, 1, 1, 1, self.mdp.num_states, 
			0.5, 1, self.rng, 'linear')
		params = net.get_params()
		zero_params = [np.zeros_like(param) for param in params]
		net.set_params(zero_params)
		self.train(net, 1000)

		np.testing.assert_allclose([[.7, 0.25], [.35, .5], [.25, 1.0], [.0, .0]]
			, self.get_q_table(net), rtol = 0.00001)

	def test_convergence_permanent_freeze(self):
		Network.TARGET_NETWORK_UPDATE_FREQUENCY = 1000000
		net = Network(self.mdp.num_actions, 1, 1, 1, self.mdp.num_states, 
			0.5, 1, self.rng, 'linear')
		params = net.get_params()
		zero_params = [np.zeros_like(param) for param in params]
		net.set_params(zero_params)
		net.clone_target()
		self.train(net, 1000)

		np.testing.assert_allclose([[.7, 0], [.0, .0], [.0, 1.0], [.0, .0]]
			, self.get_q_table(net), rtol = 0.00001)

	def test_convergence_frequent_freeze(self):
		Network.TARGET_NETWORK_UPDATE_FREQUENCY = 2
		net = Network(self.mdp.num_actions, 1, 1, 1, self.mdp.num_states, 
			0.5, 1, self.rng, 'linear')
		params = net.get_params()
		zero_params = [np.zeros_like(param) for param in params]
		net.set_params(zero_params)
		net.clone_target()
		self.train(net, 1000)

		np.testing.assert_allclose([[.7, 0.25], [.35, .5], [.25, 1.0], [.0, .0]]
			, self.get_q_table(net), rtol = 0.00001)

	def test_convergence_one_freeze(self):
		Network.TARGET_NETWORK_UPDATE_FREQUENCY = 501
		net = Network(self.mdp.num_actions, 1, 1, 1, self.mdp.num_states, 
			0.5, 1, self.rng, 'linear')
		params = net.get_params()
		zero_params = [np.zeros_like(param) for param in params]
		net.set_params(zero_params)
		net.clone_target()
		self.train(net, 1000)

		np.testing.assert_allclose([[.7, .0], [.35, .5], [.0, 1.0], [.0, .0]]
			, self.get_q_table(net), rtol = 0.00001)

	def test_convergence_random_initialization(self):
		Network.TARGET_NETWORK_UPDATE_FREQUENCY = 1
		net = Network(self.mdp.num_actions, 1, 1, 1, self.mdp.num_states, 
			0.5, 1, self.rng, 'linear')
		self.train(net, 1000)

		np.testing.assert_allclose([[.7, .25], [.35, .5], [.25, 1.0]]
			, self.get_q_table(net)[:-1, :], rtol = 0.00001)

	def test_convergence_clip_error(self):
		Network.MAX_ERROR = 0.05
		Network.TARGET_NETWORK_UPDATE_FREQUENCY = 2
		net = Network(self.mdp.num_actions, 1, 1, 1, self.mdp.num_states, 
			0.5, 1, self.rng, 'linear')
		params = net.get_params()
		zero_params = [np.zeros_like(param) for param in params]
		net.set_params(zero_params)
		net.clone_target()
		self.train(net, 1000)

		np.testing.assert_allclose([[.7, 0.25], [.35, .5], [.25, 1.0], [.0, .0]]
			, self.get_q_table(net), rtol = 0.00001)

	def test_updates_no_freeze(self):
		Network.TARGET_NETWORK_UPDATE_FREQUENCY = 0
		Network.MIN_SQR_GRAD = 1.0
		Network.LEARNING_RATE = 1.0
		Network.GRAD_MOMENTUM = 1.0
		Network.SQR_GRAD_MOMENTUM = 1.0
		net = Network(self.mdp.num_actions, 1, 1, 1, self.mdp.num_states, 
			0.5, 1, self.rng, 'linear')
		params = net.get_params()
		zero_params = [np.zeros_like(param) for param in params]
		net.set_params(zero_params)

		mdp = self.mdp
		states = np.zeros((1, 2, 1, mdp.num_states)).astype(theano.config.floatX)

		# Depart left:
		cstate = mdp.states[0]
		action_index = 0
		reward, nstate, terminal = mdp.act(cstate, action_index)
		states[:, 0, :, :] = cstate
		states[:, 1, :, :] = nstate
		net.train_one_minibatch(states, mdp.actions[action_index], reward, terminal)
		np.testing.assert_allclose([[.7, 0], [0, 0], [0, 0], [0, 0]]
			, self.get_q_table(net), rtol = 0.00001)

		# Depart right:
		cstate = mdp.states[-2]
		action_index = 1
		reward, nstate, terminal = mdp.act(cstate, action_index)
		states[:, 0, :, :] = cstate
		states[:, 1, :, :] = nstate
		net.train_one_minibatch(states, mdp.actions[action_index], reward, terminal)
		np.testing.assert_allclose([[.7, 0], [0, 0], [0, 1], [0, 0]]
			, self.get_q_table(net), rtol = 0.00001)

		# Move into leftmost state
		cstate = mdp.states[1]
		action_index = 0
		reward, nstate, terminal = mdp.act(cstate, action_index)
		states[:, 0, :, :] = cstate
		states[:, 1, :, :] = nstate
		net.train_one_minibatch(states, mdp.actions[action_index], reward, terminal)
		np.testing.assert_allclose([[.7, 0], [0.35, 0], [0, 1],[0, 0]]
			, self.get_q_table(net), rtol = 0.00001)

if __name__ == '__main__':
	unittest.main()