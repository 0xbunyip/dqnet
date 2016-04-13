import numpy as np
import time

class Experience:
	"""docstring for Experience"""
	def __init__(self, replay_mem_size, frame_height, frame_width, frames_per_state):
		self.replay_mem_size = replay_mem_size
		self.top = 0
		self.len = 0
		self.height = frame_height
		self.width = frame_width
		self.frames_per_state = frames_per_state
		self.obs = np.zeros((replay_mem_size, frame_height, frame_width), dtype = np.uint8)
		self.action = np.zeros((replay_mem_size, 1), dtype = np.uint8)
		self.reward = np.zeros((replay_mem_size, 1), dtype = np.int32)
		self.terminal = np.zeros((replay_mem_size, 1), dtype = np.bool_)

	def update_current_observation(self, obs, is_terminal):
		self.obs[self.top, :, :] = obs
		self.terminal[self.top] = is_terminal

	def add_action_reward(self, action, reward):
		self.action[self.top] = action
		self.reward[self.top] = reward

		self.top = (self.top + 1) % self.replay_mem_size
		if self.len < self.replay_mem_size:
			self.len += 1

	def get_last_state(self):
		assert self.len >= self.frames_per_state
		return self.obs.take(np.arange(self.top - self.frames_per_state, self.top), axis = 0, mode = 'wrap')

	def get_last_experience(self):
		assert self.len >= self.frames_per_state
		state = self.get_last_state()
		action = self.action[self.top - 1]
		reward = self.reward[self.top - 1]
		terminal = self.terminal[self.top - 1]
		return (state, action, reward, terminal)

	def get_random_minibatch(self, mbsize):
		assert self.len >= self.frames_per_state
		next_state = np.zeros((mbsize, self.frames_per_state, self.height, self.width), dtype = np.uint8)
		state = np.zeros((mbsize, self.frames_per_state, self.height, self.width), dtype = np.uint8)
		action = np.zeros((mbsize, 1), dtype = np.uint8)
		reward = np.zeros((mbsize, 1), dtype = np.int32)
		terminal = np.zeros((mbsize, 1), dtype = np.bool_)

		#print "Top =", self.top
		cnt = 0
		while cnt < mbsize:
			start_id = np.random.randint(1, self.len - self.frames_per_state + 1) + self.top
			end_id = start_id + self.frames_per_state - 1
			#print start_id, end_id, self.terminal.take(np.arange(start_id - 1, end_id - 1), mode = 'wrap')
			if not np.any(self.terminal.take(np.arange(start_id - 1, end_id - 1), mode = 'wrap')):				
				state[cnt, ...] = self.obs.take(np.arange(start_id - 1, end_id), axis = 0, mode = 'wrap')
				action[cnt] = self.action.take(end_id - 1, mode = 'wrap')
				reward[cnt] = self.reward.take(end_id - 1, mode = 'wrap')
				terminal[cnt] = self.terminal.take(end_id - 1, mode = 'wrap')
				next_state[cnt, ...] = self.obs.take(np.arange(start_id, end_id + 1), axis = 0, mode = 'wrap')
				cnt += 1
		return (state, action, reward, terminal, next_state)

def trivial_test():
	np.random.seed(123)
	height = 2
	width = 3
	replay_mem_size = 5
	frames_per_state = 2
	exp = Experience(replay_mem_size, height, width, frames_per_state)
	for i in xrange(8):
		obs = (i + 1) * np.ones((height, width), dtype = np.uint8)
		action = np.random.randint(4)
		reward = np.random.randint(10) * (2 * np.random.randint(2) - 1)
		terminal = (np.random.randint(5) == 0)

		exp.update_current_observation(obs, terminal)
		exp.add_action_reward(action, reward)

		print "After iteration #%d\nExp #%d:" % (i, i)
		#print exp.get_last_experience()
		print exp.obs, exp.action, exp.reward, exp.terminal

	for i in xrange(5):
		print "Get minibatch %i" % i
		print exp.get_random_minibatch(2)

def max_len_test():
	np.random.seed(123)
	height = 84
	width = 84
	frames_per_state = 10
	replay_mem_size1 = 10
	replay_mem_size2 = 100
	exp1 = Experience(replay_mem_size1, height, width, frames_per_state)
	exp2 = Experience(replay_mem_size2, height, width, frames_per_state)
	for i in xrange(20):
		obs = np.uint8(np.random.randint(0, 256, (height, width)))
		action = np.random.randint(8)
		reward = np.random.randint(10) * (2 * np.random.randint(2) - 1)
		terminal = (np.random.randint(10) == 0)

		exp1.update_current_observation(obs, terminal)
		exp1.add_action_reward(action, reward)

		exp2.update_current_observation(obs, terminal)
		exp2.add_action_reward(action, reward)

	obs = exp2.obs.take(np.arange(exp2.top - replay_mem_size1, exp2.top), axis = 0, mode = 'wrap')
	action = exp2.action.take(np.arange(exp2.top - replay_mem_size1, exp2.top), mode = 'wrap').reshape(-1, 1)
	reward = exp2.reward.take(np.arange(exp2.top - replay_mem_size1, exp2.top), mode = 'wrap').reshape(-1, 1)
	terminal = exp2.terminal.take(np.arange(exp2.top - replay_mem_size1, exp2.top), mode = 'wrap').reshape(-1, 1)

	print "obs equal", np.allclose(exp1.obs, obs)
	print "action equal", np.allclose(exp1.action, action)
	print "reward equal", np.allclose(exp1.reward, reward)
	print "terminal equal", np.allclose(exp1.terminal, terminal)

def speed_test():
	np.random.seed(123)
	height = 84
	width = 84
	frames_per_state = 4
	replay_mem_size = 1000
	exp = Experience(replay_mem_size, height, width, frames_per_state)
	start_time = time.time()
	n = 50000
	for i in xrange(n):
		obs = np.uint8(np.random.randint(0, 256, (height, width)))
		action = np.random.randint(8)
		reward = np.random.randint(10) * (2 * np.random.randint(2) - 1)
		terminal = (np.random.randint(10) == 0)

		exp.update_current_observation(obs, terminal)
		exp.add_action_reward(action, reward)
		if i % (n / 10) == 0:
			print "Done %d%%" % (i * 100 / n)

	elapsed = time.time() - start_time
	print "Per update", elapsed / n

	m = 10000
	start_time = time.time()
	for i in xrange(m):
		a, b, c, d, e = exp.get_random_minibatch(32)
		if i % (m / 10) == 0:
			print "Done %d%%" % (i * 100 / m)
	elapsed = time.time() - start_time
	print "Per access", elapsed / n

def main():
	speed_test()

if __name__ == '__main__':
	main()