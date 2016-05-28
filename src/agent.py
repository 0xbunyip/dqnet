import numpy as np
import cPickle
from experience import Experience
from network import Network
import matplotlib.pyplot as plt

class Agent:
	"""docstring for Agent"""

	HISTORY = 4
	DISCOUNT = 0.95
	FINAL_EXPLORE = 0.1
	EXPLORE_FRAMES = 1000000
	INIT_EXPLORE = 1.0
	MINIBATCH_SIZE = 32
	REPLAY_SIZE = 500000
	REPLAY_START = 50000
	UPDATE_FREQ = 4
	VALID_SIZE = 2048

	def __init__(self, num_action, frame_height, frame_width, rng, network_type
				, network_file = None, num_ignore = 0, exp_file = None):
		self.rng = rng
		self.num_action = num_action
		self.mbsize = Agent.MINIBATCH_SIZE
		self.validate_size = Agent.VALID_SIZE
		self.num_train_obs = 0
		self.network_type = network_type
		self.eps_decay = (Agent.FINAL_EXPLORE - Agent.INIT_EXPLORE) \
						/ Agent.EXPLORE_FRAMES

		self.validate_states = None
		if exp_file is not None:
			with open(exp_file, 'rb') as f:
				npz = np.load(exp_file)
				self.num_train_obs = np.sum(npz['num_train_obs'])
				self.validate_states = npz['validate_states']
				self.exp_train = Experience(Agent.REPLAY_SIZE, frame_height
										, frame_width, Agent.HISTORY, rng
										, npz)
		else:
			self.exp_train = Experience(Agent.REPLAY_SIZE, frame_height
										, frame_width, Agent.HISTORY, rng)
		self.exp_eval = Experience(Agent.HISTORY, frame_height, frame_width
									, Agent.HISTORY, rng)

		self.network = Network(num_action, self.mbsize, Agent.HISTORY, 
								frame_height, frame_width, Agent.DISCOUNT
								, Agent.UPDATE_FREQ, rng, self.network_type
								, network_file, num_ignore)

		print self.num_train_obs

	def get_action(self, obs, eps = 0.0, evaluating = False):
		exp = self.exp_eval if evaluating else self.exp_train
		if not exp.can_get_state():
			random_action = self.rng.randint(self.num_action)
			# print "Starting episode, not enough frame (%d), action = %d" \
			# 		% (exp.obs_episode + 1, random_action)
			return random_action, True

		if not evaluating:
			if self.num_train_obs < Agent.REPLAY_START:
				random_action = self.rng.randint(self.num_action)
				# print "Start training, not enough experience (%d), "
				# 	"action = %d" % (self.num_train_obs + 1, random_action)
				return random_action, True
			eps = Agent.INIT_EXPLORE + self.eps_decay * \
					min(Agent.EXPLORE_FRAMES, self.num_train_obs + 1)
			# if self.num_train_obs % 1000 == 0:
			# 	print "num_train_obs, eps =", self.num_train_obs, eps

		if self.rng.rand() < eps:
			random_action = self.rng.randint(self.num_action)
			# print "Uniform random action (obs = %d, eps = %.3f), "
			# 		"action = %d" % (self.num_train_obs + 1, eps, random_action)
			return random_action, True

		state = exp.get_state(obs)
		# self._plot_state(state)
		action = self.network.get_action(state)
		# print "Greedy action = %d" % (action)
		return action, False

	def _plot_state(self, state):
		for i in xrange(state.shape[0]):
			plt.subplot(1, state.shape[0], i + 1)
			plt.imshow(state[i, :, :], interpolation = 'none', cmap = "gray")
			plt.grid(color = 'r', linestyle = '-', linewidth = .4)
		plt.show()

	def add_experience(self, obs, is_terminal, action, reward, evaluating = False):
		exp = self.exp_eval
		if not evaluating:
			exp = self.exp_train
			self.num_train_obs += 1
		exp.add_experience(obs, is_terminal, action, reward)

		if self.num_train_obs == Agent.REPLAY_START \
								and self.validate_states is None:
			print "Collect validation states"
			self.validate_states, _, _, _ = \
				self.exp_train.get_random_minibatch(self.validate_size)

		if not evaluating and self.num_train_obs >= Agent.REPLAY_START \
			and (self.num_train_obs % Agent.UPDATE_FREQ == 0):
			self._train_one_minibatch()

	def get_validate_values(self):
		assert self.validate_states is not None
		sum_action_values = 0.0
		for i in xrange(0, self.validate_size, self.mbsize):
			states_minibatch = self.validate_states[i : \
				min(self.validate_size, i + self.mbsize), ...]

			sum_action_values += np.sum(
						self.network.get_max_action_values(states_minibatch))
		return sum_action_values / self.validate_size

	def dump_network(self, file_name):
		self.network.dump(file_name)

	def dump_exp(self, file_name):
		self.exp_train.dump(file_name, self.num_train_obs, self.validate_states)

	def _train_one_minibatch(self):
		states, action, reward, terminal = \
								self.exp_train.get_random_minibatch(self.mbsize)

		loss = self.network.train_one_minibatch(states, action, reward, terminal)
