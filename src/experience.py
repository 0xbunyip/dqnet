import numpy as np
import time

class Experience:
	"""docstring for Experience"""
	def __init__(self, replay_mem_size, frame_height, frame_width, frames_per_state, rng):
		self.rng = rng
		self.replay_mem_size = replay_mem_size
		self.top = 0
		self.len = 0
		self.obs_episode = 0
		self.height = frame_height
		self.width = frame_width
		self.frames_per_state = frames_per_state
		self.obs = np.zeros((replay_mem_size, frame_height, frame_width), dtype = np.uint8)
		self.action = np.zeros((replay_mem_size, 1), dtype = np.uint8)
		self.reward = np.zeros((replay_mem_size, 1), dtype = np.int32)
		self.terminal = np.zeros((replay_mem_size, 1), dtype = np.bool_)
		self.return_state = np.zeros((frames_per_state, frame_height, frame_width), dtype = np.uint8)

	def add_experience(self, obs, is_terminal, action, reward):
		self.obs[self.top, :, :] = obs
		self.terminal[self.top] = is_terminal
		self.action[self.top] = action
		self.reward[self.top] = reward
		
		self.obs_episode += 1
		if is_terminal:
			self.obs_episode = 0

		self.top = (self.top + 1) % self.replay_mem_size
		if self.len < self.replay_mem_size:
			self.len += 1

	def can_get_state(self):
		return self.obs_episode + 1 >= self.frames_per_state

	def get_state(self, obs):
		assert self.len + 1 >= self.frames_per_state
		self.return_state[-1, :, :] = obs
		self.return_state[:-1, :, :] = self.obs.take(np.arange(self.top - self.frames_per_state + 1, self.top), axis = 0, mode = 'wrap')
		return self.return_state

	def get_random_minibatch(self, mbsize):
		assert self.len >= self.frames_per_state
		states = np.zeros((mbsize, self.frames_per_state + 1, self.height, self.width), dtype = np.uint8)
		action = np.zeros((mbsize, 1), dtype = np.uint8)
		reward = np.zeros((mbsize, 1), dtype = np.int32)
		terminal = np.zeros((mbsize, 1), dtype = np.bool_)

		cnt = 0
		while cnt < mbsize:
			start_id = self.rng.randint(1, self.len - self.frames_per_state + 1, mbsize - cnt) + self.top * (self.len >= self.replay_mem_size)
			end_id = start_id + self.frames_per_state - 1
			not_terminal = [not np.any(self.terminal.take(np.arange(i - 1, j - 1), axis = 0, mode = 'wrap')) 
								for i, j in zip(start_id, end_id)]
			num_ok = np.sum(not_terminal)								
			ids = np.asarray([range(start_id[i] - 1, end_id[i] + 1) for i, j in enumerate(not_terminal) if j == True], dtype = np.int32)
			
			states[cnt : cnt + num_ok, ...] = self.obs.take(ids.ravel(), axis = 0, mode = 'wrap')\
							.reshape(num_ok, self.frames_per_state + 1, self.height, self.width)
			tmp = self.action.take(end_id[not_terminal] - 1, mode = 'wrap')
			action[cnt : cnt + num_ok] = self.action.take(end_id[not_terminal] - 1, mode = 'wrap').reshape(-1, 1)
			reward[cnt : cnt + num_ok] = self.reward.take(end_id[not_terminal] - 1, mode = 'wrap').reshape(-1, 1)
			terminal[cnt : cnt + num_ok] = self.terminal.take(end_id[not_terminal] - 1, mode = 'wrap').reshape(-1, 1)
			cnt += np.sum(num_ok)
		return (states, action, reward, terminal)
