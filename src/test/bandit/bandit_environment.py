import numpy as np
import cv2
from bandit_game import BanditGame

class Environment:
	"""docstring for Environment"""

	STEPS_PER_EPOCH = 10000
	FRAMES_SKIP = 1
	ORIGINAL_HEIGHT = 2
	ORIGINAL_WIDTH = 2
	FRAME_HEIGHT = 2
	FRAME_WIDTH = 2

	def __init__(self, rng, display_screen = False):
		self.api = BanditGame(Environment.ORIGINAL_HEIGHT, Environment.ORIGINAL_WIDTH, rng)
		self.agent = None
		self.minimal_actions = self.api.getMinimalActionSet()
		self.merge_frame = np.zeros((1, Environment.ORIGINAL_HEIGHT, Environment.ORIGINAL_WIDTH), dtype = np.uint8)
		self.merge_id = 0
		print self.minimal_actions

	def hook_agent(self, agent):
		self.agent = agent

	def get_action_count(self):
		return len(self.minimal_actions)

	def train_agent(self, epoch_count):
		num_test = 100
		before_reward = 0
		# before_reward, _ = self.test(num_test)

		obs = np.zeros((Environment.FRAME_HEIGHT, Environment.FRAME_WIDTH), dtype = np.uint8)
		for epoch in xrange(epoch_count):
			steps_left = Environment.STEPS_PER_EPOCH

			print "============================================"
			print "Epoch #%d" % epoch
			episode = 0
			while steps_left > 0:
				steps_left -= self._run_episode(steps_left, obs)
				if steps_left == 0 or episode % 1000 == 0:
					print "Finished episode #%d, steps_left = %d" % (episode, steps_left)
				episode += 1
			# epoch_reward, _ = self.test(num_test)
			# print "Finsihed epoch #%d, reward per episode = %.3f" % (epoch, epoch_reward)

		print "Number of frame seen:", self.agent.obs_count

		after_reward, action_list = self.test(num_test)
		print "BEFORE TRAINING: reward per episode =", before_reward
		print "AFTER TRAINING: reward per episode =", after_reward		
		print "Action list =", action_list

	def _run_episode(self, steps_left, obs):
		self.api.reset_game()		
		starting_lives = self.api.lives()
		is_terminal = False
		step_count = 0

		while step_count < steps_left and not is_terminal:
			self._get_screen(obs)
			action_id = self.agent.get_action(obs)
			
			reward = self._repeat_action(self.minimal_actions[action_id])
			is_terminal = self.api.game_over() or (self.api.lives() < starting_lives) or (step_count + 1 >= steps_left)
			self.agent.add_experience(obs, is_terminal, action_id, reward)

			# print "Step #%d, obs =" % step_count
			# print obs
			# print "Action = %d" % self.minimal_actions[action_id]
			step_count += 1
		return step_count

	def _repeat_action(self, action):
		reward = 0
		for _ in xrange(Environment.FRAMES_SKIP):
			reward += self.api.act(action)
		return reward

	def _get_screen(self, resized_frame):
		self.merge_id = (self.merge_id + 1) % 1
		self.api.getScreenGrayscale(self.merge_frame[self.merge_id, :])
		self._resize_frame(self.merge_frame.max(axis = 0), resized_frame)
				
	def _resize_frame(self, src_frame, dst_frame):
		return cv2.resize(src = src_frame, dst = dst_frame,
						dsize = (Environment.FRAME_WIDTH, Environment.FRAME_HEIGHT), 
						interpolation = cv2.INTER_LINEAR)

	def test(self, num_episode, max_episode_length = 1000, print_each_step = False):	
		total_reward = 0.0
		obs = np.zeros((Environment.FRAME_HEIGHT, Environment.FRAME_WIDTH), dtype = np.uint8)
		is_terminal = False
		action_list = np.zeros((len(self.minimal_actions), 1))

		for episode in range(num_episode):
			self.api.reset_game()
			episode_reward = 0
			step_cnt = 0
			while not self.api.game_over() and step_cnt < max_episode_length:
				self._get_screen(obs)
				action_id = self.agent.get_action(obs, e_greedy = False, print_Q = True)
				action_list[self.minimal_actions[action_id]] += 1
				reward = self._repeat_action(self.minimal_actions[action_id])

				if print_each_step:
					print "Action = %d, Obs =" % self.minimal_actions[action_id]
					print obs.astype(dtype = np.int32) - BanditGame.MAX_MEAN - BanditGame.MAX_VAR
					print "Reward =", reward
					print "\n"

				is_terminal = self.api.game_over() or (step_cnt + 1 >= max_episode_length)
				self.agent.add_experience(obs, is_terminal, action_id, reward, testing = True)
				
				episode_reward += reward
				step_cnt += 1

			#print("Episode %d ended with score: %d" % (episode, episode_reward))
			total_reward += episode_reward
		return total_reward / num_episode, action_list
		# print "Average reward per episode of %d episodes = %.5f" % (num_episode, total_reward / num_episode)
