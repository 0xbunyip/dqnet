import numpy as np
import cPickle
from experience import Experience
from network import Network

class Agent:
	"""docstring for Agent"""

	REPLAY_MEMORY_SIZE = 1000
	REPLAY_START_SIZE = 100
	AGENT_HISTORY_LENGTH = 1

	UPDATE_FREQUENCY = 4
	TARGET_NETWORK_UPDATE_FREQUENCY = 500

	DISCOUNT_FACTOR = 0.99

	INITIAL_EXPLORATION = 0.2
	FINAL_EXPLORATION = 0.2
	FINAL_EXPLORATION_FRAME = 3000

	MINIBATCH_SIZE = 32
	VALIDATION_SET_SIZE = 32

	def __init__(self, num_action, frame_height, frame_width, rng, load_network_from = None):
		self.rng = rng
		self.num_action = num_action
		self.mbsize = Agent.MINIBATCH_SIZE
		self.validate_size = Agent.VALIDATION_SET_SIZE
		self.num_train_obs = 0
		self.obs_episode = 0
		self.eps_decay_rate = (Agent.FINAL_EXPLORATION - Agent.INITIAL_EXPLORATION) / Agent.FINAL_EXPLORATION_FRAME

		self.validate_states = None
		self.exp_train = Experience(Agent.REPLAY_MEMORY_SIZE, frame_height, frame_width, Agent.AGENT_HISTORY_LENGTH, rng)
		self.exp_eval = Experience(Agent.AGENT_HISTORY_LENGTH, frame_height, frame_width, Agent.AGENT_HISTORY_LENGTH, rng)

		if load_network_from is None:
			self.network = Network(num_action, self.mbsize, Agent.AGENT_HISTORY_LENGTH, frame_height, frame_width, Agent.DISCOUNT_FACTOR, rng)
			self.tnetwork = Network(num_action, self.mbsize, Agent.AGENT_HISTORY_LENGTH, frame_height, frame_width, Agent.DISCOUNT_FACTOR, rng)
			self.network.compile_train_function(self.tnetwork)
		else:
			with open(load_network_from) as f:
				self.network = cPickle.load(f)
				self.tnetwork = cPickle.load(f)
				# CHECK NETWORK LOADING
				# with open('eval_last.txt', 'w') as f2:
				# 	params = self.network.get_params()
				# 	for param in params:
				# 		f2.write(str(np.round(param, 4).tolist()) + '\n')

	def get_action(self, obs, eps = 0.0, evaluating = False):
		random_action = self.rng.randint(self.num_action)
		if self.obs_episode + 1 < Agent.AGENT_HISTORY_LENGTH:
			# print "Starting episode, not enough frame (%d), action = %d" % (self.obs_episode + 1, random_action)
			return random_action, True

		exp = self.exp_eval
		if not evaluating:
			exp = self.exp_train
			if self.num_train_obs < Agent.REPLAY_START_SIZE:
				# print "Start training, not enough experience (%d), action = %d" % (self.num_train_obs + 1, random_action)
				return self.rng.randint(self.num_action), True
			eps = Agent.INITIAL_EXPLORATION + self.eps_decay_rate * min(Agent.FINAL_EXPLORATION_FRAME, self.num_train_obs + 1)

		if self.rng.rand() < eps:
			# print "Uniform random action (obs = %d, eps = %.3f), action = %d" % (self.num_train_obs + 1, eps, random_action)
			return self.rng.randint(self.num_action), True

		action = self.network.get_action(exp.get_state(obs))
		# print "Greedy action = %d" % (action)
		return action, False

	def add_experience(self, obs, is_terminal, action, reward, evaluating = False):
		exp = self.exp_eval
		if not evaluating:
			exp = self.exp_train
			self.num_train_obs += 1
		exp.add_experience(obs, is_terminal, action, reward)

		if self.num_train_obs == Agent.REPLAY_START_SIZE:
			print "Collect validation states"
			self.validate_states, _, _, _ = self.exp_train.get_random_minibatch(self.validate_size)

		self.obs_episode += 1
		if is_terminal:
			self.obs_episode = 0
		if not evaluating and (self.num_train_obs % Agent.UPDATE_FREQUENCY == 0) and self.num_train_obs >= Agent.REPLAY_START_SIZE:
			self._train_one_minibatch()

	def get_validate_values(self):
		assert self.validate_states is not None
		sum_action_values = 0.0
		for i in xrange(0, self.validate_size, self.mbsize):
			states_minibatch = self.validate_states[i : min(self.validate_size, i + self.mbsize), ...]
			max_action_values = self.network.get_max_action_values(states_minibatch)
			sum_action_values += np.sum(max_action_values)
		return sum_action_values / self.validate_size

	def _train_one_minibatch(self):
		states, action, reward, terminal = self.exp_train.get_random_minibatch(self.mbsize)
		if self.num_train_obs % Agent.TARGET_NETWORK_UPDATE_FREQUENCY == 0:
			# print "\tClone network at obs_count =", self.num_train_obs
			self._clone_network()
		loss = self.network.train_one_minibatch(self.tnetwork, states, action, reward, terminal)

	def _clone_network(self):
		self.tnetwork.set_params(self.network.get_params())