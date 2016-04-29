import numpy as np
from experience import Experience
from grid_network import Network

class Agent:
	"""docstring for Agent"""

	REPLAY_MEMORY_SIZE = 50000
	REPLAY_START_SIZE = 100
	AGENT_HISTORY_LENGTH = 1

	UPDATE_FREQUENCY = 4
	TARGET_NETWORK_UPDATE_FREQUENCY = 5000

	DISCOUNT_FACTOR = 0.99

	INITIAL_EXPLORATION = 1.0
	FINAL_EXPLORATION = 0.1
	FINAL_EXPLORATION_FRAME = 100000

	MINIBATCH_SIZE = 32

	def __init__(self, num_action, frame_height, frame_width, validate_size, rng):
		self.rng = rng
		self.num_action = num_action
		self.mbsize = Agent.MINIBATCH_SIZE
		self.validate_size = validate_size
		self.num_train_obs = 0
		self.obs_episode = 0
		self.eps_decay_rate = (Agent.FINAL_EXPLORATION - Agent.INITIAL_EXPLORATION) / Agent.FINAL_EXPLORATION_FRAME

		self.validate_states = None
		# self.validate_states_origin = None
		self.exp_train = Experience(Agent.REPLAY_MEMORY_SIZE, frame_height, frame_width, Agent.AGENT_HISTORY_LENGTH, rng)
		self.exp_eval = Experience(Agent.AGENT_HISTORY_LENGTH, frame_height, frame_width, Agent.AGENT_HISTORY_LENGTH, rng)

		self.network = Network(num_action, self.mbsize, Agent.AGENT_HISTORY_LENGTH, frame_height, frame_width, Agent.DISCOUNT_FACTOR)
		self.tnetwork = Network(num_action, self.mbsize, Agent.AGENT_HISTORY_LENGTH, frame_height, frame_width, Agent.DISCOUNT_FACTOR)
		self.network.compile_train_function(self.tnetwork)

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
			self.validate_states, _, _, _, _ = self.exp_train.get_random_minibatch(self.validate_size)
			# self.validate_states_origin = self.validate_states.copy()
			print "Before training, average validate action values = %.3f" % (self.get_validate_values())

		self.obs_episode += 1
		if is_terminal:
			self.obs_episode = 0
		if not evaluating and (self.num_train_obs % Agent.UPDATE_FREQUENCY == 0) and self.num_train_obs >= Agent.REPLAY_START_SIZE:
			self._train_one_minibatch()

	def get_validate_values(self):
		assert self.validate_states is not None
		# assert np.allclose(self.validate_states, self.validate_states_origin)
		sum_action_values = 0.0
		for i in xrange(0, self.validate_size, self.mbsize):
			last_id = min(self.validate_size, i + self.mbsize)
			first_id = max(0, last_id - self.mbsize)
			states_minibatch = self.validate_states[first_id : last_id, ...]
			max_action_values = self.network.get_max_action_values(states_minibatch)

			if i + self.mbsize > self.validate_size:
				# print max_action_values[-(self.validate_size - i):].shape
				sum_action_values += np.sum(max_action_values[-(self.validate_size - i):])
			else:
				# print max_action_values.shape
				sum_action_values += np.sum(max_action_values)
		return sum_action_values / self.validate_size

	def _train_one_minibatch(self):
		state, action, reward, terminal, next_state = self.exp_train.get_random_minibatch(self.mbsize)
		# print "Train one minibatch ="
		# print "Current state =\n", state
		# print "Action =", action
		# print "Reward =", reward
		# print "Terminal =", terminal
		# print "Next state =\n", next_state
		if self.num_train_obs % Agent.TARGET_NETWORK_UPDATE_FREQUENCY == 0:
			# print "\tClone network at obs_count =", self.num_train_obs
			self._clone_network()
		loss = self.network.train_one_minibatch(self.tnetwork, state, action, reward, terminal, next_state)

	def _clone_network(self):
		self.tnetwork.set_params(self.network.get_params())