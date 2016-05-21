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
		self.terminal_lol = False # Most recent episode ended on a loss of life
		self.max_start_nullops = Environment.MAX_NO_OP
		self.death_ends_episode = True

	def get_action_count(self):
		return len(self.minimal_actions)

	def train(self, agent):
		self.agent = agent
		self._open_log_files(agent)
		obs = np.zeros((self.height, self.width), dtype = np.uint8)
		epoch_count = Environment.EPOCH_COUNT

		for epoch in xrange(epoch_count):
			self.terminal_lol = False
			steps_left = Environment.STEPS_PER_EPOCH

			print "\n" + "=" * 50
			print "Epoch #%d" % (epoch + 1)
			episode = 0
			train_start = time.time()
			while steps_left > 0:
				_, num_step = self.run_episode(steps_left, False)
				# num_step, _ = self._run_episode(agent, steps_left, obs)
				steps_left -= num_step
				episode += 1
				if steps_left == 0 or episode % 10 == 0:
					print "Finished episode #%d, steps_left = %d" % (episode, steps_left)
			train_end = time.time()

			avg_validate_values = agent.get_validate_values()
			eval_values = self.evaluate(agent)
			test_end = time.time()

			train_time = train_end - train_start
			test_time = test_end - train_end
			print "Finished epoch #%d, episode trained = %d, validate values = %.3f, train time = %.0fs, test time = %.0fs, evaluate reward = %.3f" \
					% (epoch + 1, episode, avg_validate_values, train_time, test_time, eval_values)
			self._update_log_files(agent, epoch + 1, episode, avg_validate_values, train_time, test_time, eval_values)
		print "Number of frame seen:", agent.num_train_obs

	def evaluate(self, agent, num_eval_episode = 30, eps = 0.05, obs = None):
		self.agent = agent
		print "\n***Start evaluating"
		if obs is None:
			obs = np.zeros((self.height, self.width), dtype = np.uint8)
		sum_reward = 0.0
		sum_step = 0.0
		for episode in xrange(num_eval_episode):
			self.terminal_lol = False
			_, step = self.run_episode(Environment.STEPS_PER_EPISODE, True)
			# step, reward = self._run_episode(agent, \
			# 	Environment.STEPS_PER_EPISODE, obs, eps, evaluating = True)
			sum_reward += self.episode_reward
			sum_step += step
			print "Finished episode %d, reward = %d, step = %d" % (episode + 1, self.episode_reward, step)
			# print "Finished episode %d, reward = %d, step = %d" % (episode + 1, reward, step)
		print "Average reward per episode = %.4f" % (sum_reward / num_eval_episode)
		print "Average step per episode = %.4f" % (sum_step / num_eval_episode)
		return sum_reward / num_eval_episode

	def run_episode(self, max_steps, testing):
		"""Run a single training episode.

		The boolean terminal value returned indicates whether the
		episode ended because the game ended or the agent died (True)
		or because the maximum number of steps was reached (False).
		Currently this value will be ignored.

		Return: (terminal, num_steps)

		"""

		self.api.reset_game()
		if Environment.MAX_NO_OP > 0:
			num_no_op = self.rng.randint(Environment.MAX_NO_OP) + self.buffer_len
			# print "Number of no-op = %d" % (num_no_op)
			for _ in xrange(num_no_op):
				self.api.act(0)

		for _ in xrange(self.buffer_len):
			self._update_buffer()
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
			f.write('Use Environment logging, screen buffer, repeat_action and train/evaluate\n')
			f.write('Use ale_experiment run_episode and _init_episode\n')
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