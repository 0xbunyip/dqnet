import numpy as np
from experience import Experience
from network import Network

class Agent:
	"""docstring for Agent"""

	REPLAY_MEMORY_SIZE = 10000
	REPLAY_START_SIZE = 500
	AGENT_HISTORY_LENGTH = 4	

	UPDATE_FREQUENCY = 4
	TARGET_NETWORK_UPDATE_FREQUENCY = 200

	DISCOUNT_FACTOR = 0.99

	def __init__(self, num_action, frame_height, frame_width, mbsize):
		self.num_action = num_action
		self.mbsize = mbsize
		self.obs_count = 0
		self.exp = Experience(Agent.REPLAY_MEMORY_SIZE, frame_height, frame_width, Agent.AGENT_HISTORY_LENGTH)
		self.network = Network(num_action, mbsize, Agent.AGENT_HISTORY_LENGTH, frame_height, frame_width, Agent.DISCOUNT_FACTOR)
		self.tnetwork = Network(num_action, mbsize, Agent.AGENT_HISTORY_LENGTH, frame_height, frame_width, Agent.DISCOUNT_FACTOR)
		self.network.compile_train_function(self.tnetwork)

	def get_latest_action(self):
		if self.obs_count < Agent.REPLAY_START_SIZE:
			return np.random.randint(self.num_action)
		return np.random.randint(self.num_action)
		
	def update_current_observation(self, obs, is_terminal):
		self.exp.update_current_observation(obs, is_terminal)

	def add_action_reward(self, action, reward):
		self.exp.add_action_reward(action, reward)
		self.obs_count += 1
		if self.obs_count >= Agent.REPLAY_START_SIZE and (self.obs_count % Agent.UPDATE_FREQUENCY == 0):
			#print "Train one minibatch at obs_count =", self.obs_count
			self._train_one_minibatch()

	def _train_one_minibatch(self):
		state, action, reward, terminal, next_state = self.exp.get_random_minibatch(self.mbsize)

		if self.obs_count % Agent.TARGET_NETWORK_UPDATE_FREQUENCY == 0:
			print "Clone network at obs_count =", self.obs_count
			self._clone_network()
		self.network.train_one_minibatch(self.tnetwork, state, action, reward, terminal, next_state)

	def _clone_network(self):
		self.tnetwork.set_params(self.network.get_params())