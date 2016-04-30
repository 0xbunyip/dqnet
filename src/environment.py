import numpy as np
import cv2
import time
import os
import cPickle
from ale_python_interface import ALEInterface

class Environment:
	"""docstring for Environment"""

	STEPS_PER_EPOCH = 2000
	STEPS_PER_EPISODE = 200
	FRAMES_SKIP = 4

	ORIGINAL_HEIGHT = 210
	ORIGINAL_WIDTH = 160
	FRAME_HEIGHT = 8
	FRAME_WIDTH = 8

	MAX_REWARD = 0
	MAX_NO_OP = 30

	def __init__(self, rom_name, rng, display_screen = False):
		self.api = ALEInterface()
		self.api.setInt('random_seed', 123)
		self.api.setBool('display_screen', display_screen)
		self.rom_name = rom_name
		self.rng = rng
		print '../rom/' + self.rom_name
		self.api.loadROM('../rom/' + self.rom_name)
		self.minimal_actions = self.api.getMinimalActionSet()
		self.merge_frame = np.zeros((2, Environment.ORIGINAL_HEIGHT, Environment.ORIGINAL_WIDTH), dtype = np.uint8)
		self.merge_id = 0
		self.max_reward = Environment.MAX_REWARD
		self.log_dir = ''

	def get_action_count(self):
		return len(self.minimal_actions)

	def train(self, agent, epoch_count):
		self._open_log_files(agent)
		obs = np.zeros((Environment.FRAME_HEIGHT, Environment.FRAME_WIDTH), dtype = np.uint8)
		for epoch in xrange(epoch_count):
			steps_left = Environment.STEPS_PER_EPOCH

			print "\n============================================"
			print "Epoch #%d" % epoch
			episode = 0
			epoch_start = time.time()
			while steps_left > 0:
				num_step, _ = self._run_episode(agent, steps_left, obs)
				steps_left -= num_step
				if steps_left == 0 or episode % 100 == 0:
					print "Finished episode #%d, steps_left = %d" % (episode, steps_left)
				episode += 1
			epoch_end = time.time()
			avg_validate_values = agent.get_validate_values()
			validate_end = time.time()

			total_train_time = epoch_end - epoch_start
			total_validate_time = validate_end - epoch_end
			print "Finished epoch #%d, episode trained = %d, validate values = %.3f, train time = %.0fs, validate time = %.0fs" \
					% (epoch, episode, avg_validate_values, total_train_time, total_validate_time)
			self._update_log_files(agent, epoch + 1, episode, avg_validate_values, total_train_time, total_validate_time)

		print "Number of frame seen:", agent.num_train_obs
		avg_reward = self.evaluate(agent, num_eval_episode = 1, obs = obs)
		print "Test average reward = %.3f" % (avg_reward)

	def evaluate(self, agent, num_eval_episode = 30, eps = 0.05, obs = None):
		print "\n***Start evaluating"
		if obs is None:
			obs = np.zeros((Environment.FRAME_HEIGHT, Environment.FRAME_WIDTH), dtype = np.uint8)
		sum_reward = 0.0
		for episode in xrange(num_eval_episode):
			_, reward = self._run_episode(agent, Environment.STEPS_PER_EPISODE, obs, eps, evaluating = True)
			sum_reward += reward
		return sum_reward / num_eval_episode

	def _run_episode(self, agent, steps_left, obs, eps = 0.0, evaluating = False):
		self.api.reset_game()		
		starting_lives = self.api.lives()
		step_count = 0
		sum_reward = 0

		if evaluating and Environment.MAX_NO_OP > 0:
			for _ in xrange(self.rng.randint(Environment.MAX_NO_OP) + 1):
				self.api.act(0)

		is_terminal = self.api.game_over() or (self.api.lives() < starting_lives)
		while step_count < steps_left and not is_terminal:
			self._get_screen(obs)
			action_id, _ = agent.get_action(obs, eps, evaluating)
			
			reward = self._repeat_action(self.minimal_actions[action_id])
			if not evaluating and self.max_reward > 0:
				reward = np.clip(reward, -self.max_reward, self.max_reward)
			is_terminal = self.api.game_over() or (self.api.lives() < starting_lives) or (step_count + 1 >= steps_left)
			agent.add_experience(obs, is_terminal, action_id, reward, evaluating)

			sum_reward += reward
			step_count += 1
		return step_count, sum_reward

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

	def _open_log_files(self, agent):
		# CREATE A FOLDER TO HOLD RESULTS
		time_str = time.strftime("_%m-%d-%H-%M", time.localtime())
		base_rom_name = os.path.splitext(os.path.basename(self.rom_name))[0]
		self.log_dir = '../run_results/' + base_rom_name + time_str

		try:
			os.stat(self.log_dir)
		except OSError:
			os.makedirs(self.log_dir)

		with open(self.log_dir + '/info.txt', 'w') as f:
			self._write_info(f, Environment)
			f.write('\n')
			self._write_info(f, agent.__class__)
			f.write('\n')
			self._write_info(f, agent.network.__class__)

		with open(self.log_dir + '/results.csv', 'w') as f:
			f.write("epoch,episode_train,validate_values,total_train_time,steps_per_second,total_validate_time\n")

	def _update_log_files(self, agent, epoch, episode, validate_values, total_train_time, total_validate_time):
		print "Updating log files"
		with open(self.log_dir + '/results.csv', 'a') as f:
			f.write("%d,%d,%.3f,%.0f,%.2f,%.0f\n" % (epoch, episode, validate_values, \
				total_train_time, total_train_time * 1.0 / Environment.STEPS_PER_EPOCH, total_validate_time))

		# with open(self.log_dir + ('/network_params_%03d' % (epoch)) + '.pkl', 'w') as f:
		# 	cPickle.dump(agent.network, f, -1)

	def _write_info(self, f, c):
		hyper_params = [attr for attr in dir(c) \
			if not attr.startswith("__") and not callable(getattr(c, attr))]
		for param in hyper_params:
			f.write(str(c.__name__) + '.' + param + ' = ' + str(getattr(c, param)) + '\n')