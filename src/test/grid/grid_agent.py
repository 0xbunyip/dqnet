import numpy as np
from experience import Experience
from grid_network import Network

class Agent:
	"""docstring for Agent"""

	REPLAY_MEMORY_SIZE = 10000
	REPLAY_START_SIZE = 100
	AGENT_HISTORY_LENGTH = 1

	UPDATE_FREQUENCY = 4
	TARGET_NETWORK_UPDATE_FREQUENCY = 500

	DISCOUNT_FACTOR = 0.99

	INITIAL_EXPLORATION = 1.0
	FINAL_EXPLORATION = 0.1
	FINAL_EXPLORATION_FRAME = 50000

	def __init__(self, num_action, frame_height, frame_width, mbsize, rng):
		self.rng = rng
		self.num_action = num_action
		self.mbsize = mbsize
		self.obs_count = 0
		self.obs_episode = 0
		self.eps_decay_rate = (Agent.FINAL_EXPLORATION - Agent.INITIAL_EXPLORATION) / Agent.FINAL_EXPLORATION_FRAME

		self.exp = Experience(Agent.REPLAY_MEMORY_SIZE, frame_height, frame_width, Agent.AGENT_HISTORY_LENGTH, rng)
		self.network = Network(num_action, mbsize, Agent.AGENT_HISTORY_LENGTH, frame_height, frame_width, Agent.DISCOUNT_FACTOR)
		self.tnetwork = Network(num_action, mbsize, Agent.AGENT_HISTORY_LENGTH, frame_height, frame_width, Agent.DISCOUNT_FACTOR)
		self.network.compile_train_function(self.tnetwork)

	def get_action(self, obs, e_greedy = True, print_Q = False):
		if self.obs_count < Agent.REPLAY_START_SIZE or self.obs_episode + 1 < Agent.AGENT_HISTORY_LENGTH:
			return self.rng.randint(self.num_action)
		if e_greedy:
			eps = Agent.INITIAL_EXPLORATION + self.eps_decay_rate * min(Agent.FINAL_EXPLORATION_FRAME, self.obs_count)
			if self.rng.rand() < eps:
				return self.rng.randint(self.num_action)
		state = self.exp.get_state(obs)
		action = self.network.get_action(state, print_Q = print_Q)
		# print "Got action %d from state" % action
		# print state
		return action

	def add_experience(self, obs, is_terminal, action, reward, testing = False):
		self.exp.add_experience(obs, is_terminal, action, reward)
		self.obs_episode += 1

		if not testing:
			self.obs_count += 1
		if is_terminal:
			self.obs_episode = 0
		if not testing and self.obs_count >= Agent.REPLAY_START_SIZE and (self.obs_count % Agent.UPDATE_FREQUENCY == 0):
			#print "Train one minibatch at obs_count =", self.obs_count
			self._train_one_minibatch()

	def _train_one_minibatch(self):
		state, action, reward, terminal, next_state = self.exp.get_random_minibatch(self.mbsize)
		# print "Train one minibatch ="
		# print "Current state =\n", state
		# print "Action =", action
		# print "Reward =", reward
		# print "Terminal =", terminal
		# print "Next state =\n", next_state
		if self.obs_count % Agent.TARGET_NETWORK_UPDATE_FREQUENCY == 0:
			print "\tClone network at obs_count =", self.obs_count
			self._clone_network()
		loss = self.network.train_one_minibatch(self.tnetwork, state, action, reward, terminal, next_state)

	def _clone_network(self):
		self.tnetwork.set_params(self.network.get_params())