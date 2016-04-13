import numpy as np
import cv2
from ale_python_interface import ALEInterface

class Environment:
	"""docstring for Environment"""

	STEPS_PER_EPOCH = 2000
	FRAMES_SKIP = 4
	ORIGINAL_HEIGHT = 210
	ORIGINAL_WIDTH = 160
	FRAME_HEIGHT = 16
	FRAME_WIDTH = 16

	def __init__(self, rom_name, display_screen = False):
		self.api = ALEInterface()
		self.api.setInt('random_seed', 123)
		self.api.setBool('display_screen', display_screen)
		self.rom_name = rom_name
		self.agent = None
		self.api.loadROM('../rom/' + self.rom_name)
		self.minimal_actions = self.api.getMinimalActionSet()
		self.merge_frame = np.zeros((2, Environment.ORIGINAL_HEIGHT, Environment.ORIGINAL_WIDTH), dtype = np.uint8)
		self.merge_id = 0
		print self.minimal_actions

	def hook_agent(self, agent):
		self.agent = agent

	def get_action_count(self):
		return len(self.minimal_actions)

	def train_agent(self, epoch_count):
		obs = np.zeros((Environment.FRAME_HEIGHT, Environment.FRAME_WIDTH), dtype = np.uint8)
		for epoch in xrange(epoch_count):
			steps_left = Environment.STEPS_PER_EPOCH

			print "============================================"
			print "Epoch #%d" % epoch
			episode = 0
			while steps_left > 0:
				steps_left -= self._run_episode(steps_left, obs)
				print "Finished episode #%d, steps_left = %d" % (episode, steps_left)
				episode += 1

		print "Number of frame seen:", self.agent.obs_count

	def _run_episode(self, steps_left, obs):
		self.api.reset_game()		
		starting_lives = self.api.lives()
		is_terminal = False
		step_count = 1

		while step_count < steps_left and not is_terminal:
			self._get_screen(obs)
			self.agent.update_current_observation(obs, is_terminal)

			action_id = self.agent.get_latest_action()
			reward = self._repeat_action(self.minimal_actions[action_id])
			self.agent.add_action_reward(action_id, reward)

			is_terminal = self.api.game_over() or (self.api.lives() < starting_lives) or (step_count + 1 >= steps_left)
			step_count += 1
		return step_count

	def _repeat_action(self, action):
		reward = 0
		for _ in xrange(Environment.FRAMES_SKIP):
			reward += self.api.act(action)
		return reward

	def _get_screen(self, resized_frame):
		self.merge_id = (self.merge_id + 1) % 2
		self.api.getScreenGrayscale(self.merge_frame[self.merge_id, :])
		return self._resize_frame(self.merge_frame.max(axis = 0), resized_frame)
				
	def _resize_frame(self, src_frame, dst_frame):
		return cv2.resize(src = src_frame, dst = dst_frame,
						dsize = (Environment.FRAME_WIDTH, Environment.FRAME_HEIGHT), 
						interpolation = cv2.INTER_LINEAR)

	def run(self):		
		for episode in range(5):
			total_reward = 0
			while not self.api.game_over():
				a = self.minimal_actions[self.agent.get_latest_action()]
				# Apply an action and get the resulting reward
				reward = self.api.act(a);
				total_reward += reward
			print("Episode %d ended with score: %d" % (episode, total_reward))
			self.api.reset_game()


