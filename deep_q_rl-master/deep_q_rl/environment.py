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
		self.num_epochs = Environment.EPOCH_COUNT
		self.epoch_length = Environment.STEPS_PER_EPOCH
		self.test_length = 125000
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
		self._open_log_files(agent)
		self.agent = agent
		for epoch in range(1, self.num_epochs + 1):
			train_start = time.time()
			train_episode, _ = self.run_epoch(epoch, self.epoch_length)
			train_stop = time.time()

			# with open(self.exp_dir + '/network_file_' + str(epoch) + \
			# 			'.pkl', 'wb') as f:
			# 	self.agent.dump(f)
			# self.agent.finish_epoch(epoch)

			test_reward = 0
			if self.test_length > 0:
				# self.agent.start_testing()
				test_episode, test_reward = self.run_epoch(epoch, self.test_length, True)
				test_stop = time.time()

				holdout_sum = self.agent.get_validate_values()
				test_reward /= float(test_episode)
				# out = "{},{},{},{},{}\n".format(epoch, self.episode_counter, 
				# 	self.total_reward, self.total_reward / float(self.episode_counter),
				# 	holdout_sum)
				# self.results_file.write(out)
				# self.results_file.flush()
				# self.agent.finish_testing(epoch)

			train_time = train_stop - train_start
			test_time = test_stop - train_stop
			self._update_log_files(agent, epoch, train_episode, \
				holdout_sum, train_time, test_time, test_reward)

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
		episode = 0
		reward = 0
		while steps_left > 0:
			prefix = "testing" if testing else "training"
			print prefix + " epoch: " + str(epoch) + " steps_left: " + str(steps_left)
			_, num_steps = self.run_episode(steps_left, testing)

			episode += 1
			reward += self.episode_reward

			steps_left -= num_steps
		return episode, reward

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
					self.api.act(0) # Null action
					self._update_buffer()

		# Make sure the screen buffer is filled at the beginning of
		# each episode...
		self.api.act(0)
		self._update_buffer()
		self.api.act(0)
		self._update_buffer()

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

		obs = np.zeros((self.height, self.width), dtype = np.uint8)
		self._get_screen(obs)
		# obs = self.get_observation()
		action, _ = self.agent.get_action(obs, 0.05, testing)
		# action = self.agent.start_episode(self.get_observation())
		num_steps = 0
		while True:
			reward = self._repeat_action(self.minimal_actions[action])
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

			self._get_screen(obs)
			# obs = self.get_observation()
			action, _ = self.agent.get_action(obs, 0.05, testing)
			# action = self.agent.step(reward, self.get_observation())
		return terminal, num_steps

	def _update_buffer(self):
		self.api.getScreenGrayscale(self.merge_frame[self.merge_id, ...])
		# print self.api.getEpisodeFrameNumber()
		self.merge_id = (self.merge_id + 1) % self.buffer_len

	def _repeat_action(self, action):
		reward = 0
		for i in xrange(self.repeat):
			reward += self.api.act(action)
			if i + self.buffer_len >= self.repeat:
				self._update_buffer()
		return reward

	def _get_screen(self, resized_frame):
		self._resize_frame(self.merge_frame.max(axis = 0), resized_frame)
				
	def _resize_frame(self, src_frame, dst_frame):
		cv2.resize(src = src_frame, dst = dst_frame,
					dsize = (self.width, self.height),
					interpolation = cv2.INTER_LINEAR)

	def _open_log_files(self, agent):
		# CREATE A FOLDER TO HOLD RESULTS
		time_str = time.strftime("_%m-%d-%H-%M", time.localtime())
		base_rom_name = os.path.splitext(os.path.basename(self.rom_name))[0]
		self.log_dir = '../run_results/' + base_rom_name + time_str
		self.network_dir = self.log_dir + '/network'

		try:
			os.stat(self.log_dir)
		except OSError:
			os.makedirs(self.log_dir)

		try:
			os.stat(self.network_dir)
		except OSError:
			os.makedirs(self.network_dir)

		with open(self.log_dir + '/info.txt', 'w') as f:
			f.write('Test env + agent + net + exp\n')
			f.write('Use Environment logging, screen buffer and repeat_action\n')
			f.write('Use ale_experiment run_episode and run_epoch\n')
			f.write(str(agent.network.network_description + '\n\n'))
			self._write_info(f, Environment)
			self._write_info(f, agent.__class__)
			self._write_info(f, agent.network.__class__)

		with open(self.log_dir + '/results.csv', 'w') as f:
			f.write("epoch,episode_train,validate_values,total_train_time,steps_per_second,total_validate_time,evaluate_reward\n")

	def _update_log_files(self, agent, epoch, episode, validate_values, total_train_time, total_validate_time, evaluate_values):
		print "Updating log files"
		with open(self.log_dir + '/results.csv', 'a') as f:
			f.write("%d,%d,%.4f,%.0f,%.4f,%.0f,%.4f\n" % (epoch, episode, validate_values, \
				total_train_time, Environment.STEPS_PER_EPOCH * 1.0 / max(1, total_train_time), 
				total_validate_time, evaluate_values))

		with open(self.network_dir + ('/%03d' % (epoch)) + '.pkl', 'wb') as f:
			agent.dump(f)

	def _write_info(self, f, c):
		hyper_params = [attr for attr in dir(c) \
			if not attr.startswith("__") and not callable(getattr(c, attr))]
		for param in hyper_params:
			f.write(str(c.__name__) + '.' + param + ' = ' + str(getattr(c, param)) + '\n')
		f.write('\n')