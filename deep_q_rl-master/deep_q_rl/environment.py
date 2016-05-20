import numpy as np
import cv2
import time
import os
from ale_python_interface import ALEInterface

class Environment:
	"""docstring for Environment"""

	BUFFER_LEN = 2
	EPOCH_COUNT = 200
	FRAMES_SKIP = 4
	FRAME_HEIGHT = 84
	FRAME_WIDTH = 84
	MAX_NO_OP = 30
	MAX_REWARD = 1
	STEPS_PER_EPISODE = 18000
	STEPS_PER_EPOCH = 50000

	def __init__(self, rom_name, rng, display_screen = False):
		self.api = ALEInterface()
		self.api.setInt('random_seed', rng.randint(333))
		self.api.setBool('display_screen', display_screen)
		self.api.setFloat('repeat_action_probability', 0.0)
		self.rom_name = rom_name
		self.rng = rng
		self.api.loadROM('../roms/' + self.rom_name)
		self.minimal_actions = self.api.getMinimalActionSet()
		self.repeat = Environment.FRAMES_SKIP
		self.buffer_len = Environment.BUFFER_LEN
		self.height = Environment.FRAME_HEIGHT
		self.width = Environment.FRAME_WIDTH

		original_width, original_height = self.api.getScreenDims()
		self.merge_frame = np.zeros((self.buffer_len
								, original_height
								, original_width)
								, dtype = np.uint8)
		self.merge_id = 0
		self.max_reward = Environment.MAX_REWARD
		self.log_dir = ''
		self.network_dir = ''

		self.ale_experiment_init()

	def ale_experiment_init(self):
		# CREATE A FOLDER TO HOLD RESULTS
		time_str = time.strftime("_%m-%d-%H-%M", time.gmtime())
		exp_dir = 'breakout' + time_str

		try:
			os.stat(exp_dir)
		except OSError:
			os.makedirs(exp_dir)

		self.exp_dir = exp_dir
		with open(self.exp_dir + '/info.txt', 'w') as f:
			f.write('Test env + agent + net + exp, use ale_experiment methods\n')
			f.write('Keep launcher.py and Environment.__init__ the same\n')

		self.results_file = open(self.exp_dir + '/results.csv', 'w', 0)
		self.results_file.write(\
			'epoch,num_episodes,total_reward,reward_per_epoch,mean_q\n')
		self.results_file.flush()

		self.num_epochs = Environment.EPOCH_COUNT
		self.epoch_length = Environment.STEPS_PER_EPOCH
		self.test_length = 100
		self.resize_method = 'scale'
		self.terminal_lol = False # Most recent episode ended on a loss of life
		self.max_start_nullops = Environment.MAX_NO_OP
		self.death_ends_episode = True

	def get_action_count(self):
		return len(self.minimal_actions)

	def train(self, agent):
		"""
		Run the desired number of training epochs, a testing epoch
		is conducted after each training epoch.
		"""
		self.agent = agent
		for epoch in range(1, self.num_epochs + 1):
			self.run_epoch(epoch, self.epoch_length)

			with open(self.exp_dir + '/network_file_' + str(epoch) + \
						'.pkl', 'wb') as f:
				self.agent.dump(f)
			# self.agent.finish_epoch(epoch)

			if self.test_length > 0:
				# self.agent.start_testing()
				self.total_reward = 0
				self.episode_counter = 0
				self.run_epoch(epoch, self.test_length, True)

				holdout_sum = self.agent.get_validate_values()
				out = "{},{},{},{},{}\n".format(epoch, self.episode_counter, 
					self.total_reward, self.total_reward / float(self.episode_counter),
					holdout_sum)
				self.results_file.write(out)
				self.results_file.flush()
				# self.agent.finish_testing(epoch)

	def run_epoch(self, epoch, num_steps, testing=False):
		""" Run one 'epoch' of training or testing, where an epoch is defined
		by the number of steps executed.  Prints a progress report after
		every trial

		Arguments:
		epoch - the current epoch number
		num_steps - steps per epoch
		testing - True if this Epoch is used for testing and not training

		"""
		self.terminal_lol = False # Make sure each epoch starts with a reset.
		steps_left = num_steps
		while steps_left > 0:
			prefix = "testing" if testing else "training"
			print prefix + " epoch: " + str(epoch) + " steps_left: " + str(steps_left)
			_, num_steps = self.run_episode(steps_left, testing)

			if testing:
				self.episode_counter += 1
				self.total_reward += self.episode_reward

			steps_left -= num_steps


	def _init_episode(self):
		""" This method resets the game if needed, performs enough null
		actions to ensure that the screen buffer is ready and optionally
		performs a randomly determined number of null action to randomize
		the initial game state."""

		if not self.terminal_lol or self.api.game_over():
			self.api.reset_game()

			if self.max_start_nullops > 0:
				random_actions = self.rng.randint(0, self.max_start_nullops+1)
				for _ in range(random_actions):
					self._act(0) # Null action

		# Make sure the screen buffer is filled at the beginning of
		# each episode...
		self._act(0)
		self._act(0)


	def _act(self, action):
		"""Perform the indicated action for a single frame, return the
		resulting reward and store the resulting screen image in the
		buffer

		"""
		reward = self.api.act(action)
		index = self.merge_id % self.buffer_len

		self.api.getScreenGrayscale(self.merge_frame[index, ...])

		self.merge_id += 1
		return reward

	def _step(self, action):
		""" Repeat one action the appopriate number of times and return
		the summed reward. """
		reward = 0
		for _ in range(self.repeat):
			reward += self._act(action)

		return reward

	def run_episode(self, max_steps, testing):
		"""Run a single training episode.

		The boolean terminal value returned indicates whether the
		episode ended because the game ended or the agent died (True)
		or because the maximum number of steps was reached (False).
		Currently this value will be ignored.

		Return: (terminal, num_steps)

		"""

		self._init_episode()
		self.episode_reward = 0

		start_lives = self.api.lives()

		obs = self.get_observation()
		action, _ = self.agent.get_action(obs, 0.05, testing)
		# action = self.agent.start_episode(self.get_observation())
		num_steps = 0
		while True:
			reward = self._step(self.minimal_actions[action])
			self.episode_reward += reward
			reward = np.clip(reward, -1, 1)
			# reward = self._step(self.min_action_set[action])

			self.terminal_lol = (self.death_ends_episode and not testing and
								 self.api.lives() < start_lives)
			terminal = self.api.game_over() or self.terminal_lol
			num_steps += 1

			if terminal or num_steps >= max_steps:
				# self.agent.end_episode(reward, terminal)
				self.agent.add_experience(obs, True, action, reward, testing)
				break
			else:
				self.agent.add_experience(obs, False, action, reward, testing)

			# if terminal or num_steps >= max_steps:
			#     self.agent.end_episode(reward, terminal)
			#     break

			obs = self.get_observation()
			action, _ = self.agent.get_action(obs, 0.05, testing)
			# action = self.agent.step(reward, self.get_observation())
		return terminal, num_steps


	def get_observation(self):
		""" Resize and merge the previous two screen images """

		assert self.merge_id >= 2
		index = self.merge_id % self.buffer_len - 1
		max_image = np.maximum(self.merge_frame[index, ...],
							   self.merge_frame[index - 1, ...])
		return self.resize_image(max_image)

	def resize_image(self, image):
		""" Appropriately resize a single image """

		if self.resize_method == 'crop':
			# resize keeping aspect ratio
			resize_height = int(round(
				float(self.height) * self.resized_width / self.width))

			resized = cv2.resize(image,
								 (self.resized_width, resize_height),
								 interpolation=cv2.INTER_LINEAR)

			# Crop the part we want
			crop_y_cutoff = resize_height - CROP_OFFSET - self.resized_height
			cropped = resized[crop_y_cutoff:
							  crop_y_cutoff + self.resized_height, :]

			return cropped
		elif self.resize_method == 'scale':
			return cv2.resize(image,
							  (self.width, self.height),
							  interpolation=cv2.INTER_LINEAR)
		else:
			raise ValueError('Unrecognized image resize method.')
