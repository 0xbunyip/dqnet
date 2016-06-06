import numpy as np
import time

class Experience:
	'''docstring for Experience'''
	def __init__(self, mem_size, frame_height, frame_width
				, history, rng, npz = None):
		self.rng = rng
		self.mem_size = mem_size
		self.top = 0
		self.len = 0
		self.obs_episode = 0
		self.height = frame_height
		self.width = frame_width
		self.history = history

		if npz is not None:
			self.top = np.sum(npz['top'])
			self.len = np.sum(npz['len'])
			self.obs = npz['obs']
			self.action = npz['action']
			self.reward = npz['reward']
			self.terminal = npz['terminal']
			self.id_list = npz['id_list'].tolist()
		else:
			self.obs = np.zeros((mem_size, frame_height, frame_width)
								, dtype = np.uint8)
			self.action = np.zeros((mem_size, 1), dtype = np.uint8)
			self.reward = np.zeros((mem_size, 1), dtype = np.float32)
			self.terminal = np.zeros((mem_size, 1), dtype = np.bool_)
			self.id_list = []

		self.return_state = np.zeros((history, frame_height
									, frame_width), dtype = np.uint8)

	def add_experience(self, obs, is_terminal, action, reward):
		self.obs[self.top, :, :] = obs
		self.terminal[self.top] = is_terminal
		self.action[self.top] = action
		self.reward[self.top] = reward
		
		self.obs_episode += 1
		if is_terminal:
			self.obs_episode = 0

		if self.len >= self.mem_size:
			if len(self.id_list) > 0 and self.top == self.id_list[0]:
				self.id_list = self.id_list[1:]

		self.top = (self.top + 1) % self.mem_size
		if self.len < self.mem_size:
			self.len += 1

		if self.len >= self.history + 1:
			i = self.top - self.history - 1
			if i < 0:
				i += self.mem_size
			j = i + self.history - 1
			if not np.any(self.terminal.take(np.arange(i, j)
											, axis = 0, mode = 'wrap')):
				self.id_list.append(i)

	def can_get_state(self):
		return self.obs_episode + 1 >= self.history

	def get_state(self, obs):
		assert self.len + 1 >= self.history
		self.return_state[-1, :, :] = obs
		self.return_state[:-1, :, :] = self.obs.take(
					np.arange(self.top - self.history + 1, self.top)
					, axis = 0, mode = 'wrap')
		return self.return_state

	def get_random_minibatch(self, mbsize):
		assert self.len >= self.history
		states = np.zeros((mbsize, self.history + 1, 
							self.height, self.width), dtype = np.uint8)
		action = np.zeros((mbsize, 1), dtype = np.uint8)
		reward = np.zeros((mbsize, 1), dtype = np.float32)
		terminal = np.zeros((mbsize, 1), dtype = np.bool_)

		rand_id = np.asarray([self.id_list[j] for j in \
							self.rng.randint(0, len(self.id_list), mbsize)]
							, dtype = np.int32)

		ids = np.asarray([range(i, i + self.history + 1) for i in rand_id]
							, dtype = np.int32)

		states[...] = self.obs.take(ids.ravel(), axis = 0, mode = 'wrap')\
						.reshape(mbsize, self.history + 1
								, self.height, self.width)
		action[...] = self.action.take(
			rand_id + self.history - 1, mode = 'wrap').reshape(-1, 1)
		reward[...] = self.reward.take(
			rand_id + self.history - 1, mode = 'wrap').reshape(-1, 1)
		terminal[...] = self.terminal.take(
			rand_id + self.history - 1, mode = 'wrap').reshape(-1, 1)

		return (states, action, reward, terminal)

	def dump(self, file_name, num_train_obs, validate_states):
		arrays = {'num_train_obs' : np.array(num_train_obs)
				, 'validate_states' : validate_states
				, 'top' : np.array(self.top)
				, 'len' : self.len
				, 'obs' : self.obs
				, 'action' : self.action
				, 'reward' : self.reward
				, 'terminal' : self.terminal
				, 'id_list' : np.asarray(self.id_list, dtype = np.int32)}
		np.savez_compressed(file_name, **arrays)

def trivial_tests():
	dataset = Experience(mem_size = 3, frame_height = 1, frame_width = 2
						, history = 2, rng = np.random.RandomState(42)
						, npz = None)

	img1 = np.array([[1, 1]], dtype='uint8')
	img2 = np.array([[2, 2]], dtype='uint8')
	img3 = np.array([[3, 3]], dtype='uint8')
	img4 = np.array([[4, 4]], dtype='uint8')
	img5 = np.array([[5, 5]], dtype='uint8')

	dataset.add_experience(img1, False, 1, 1)
	dataset.add_experience(img2, False, 2, 2)

	# print "last =", dataset.get_state(img3)
	dataset.add_experience(img3, True, 3, 3)
	# print "random =", dataset.get_random_minibatch(1)

	# print "last =", dataset.get_state(img4)
	dataset.add_experience(img4, False, 4, 4)
	# print "random =", dataset.get_random_minibatch(2)

	dataset.add_experience(img5, False, 5, 5)

def max_size_tests():
	dataset1 = Experience(mem_size = 10, frame_height = 4, frame_width = 3
						, history = 4, rng = np.random.RandomState(42)
						, npz = None)
	dataset2 = Experience(mem_size = 1000, frame_height = 4, frame_width = 3
						, history = 4, rng = np.random.RandomState(42)
						, npz = None)
	for i in range(100):
		img = np.random.randint(0, 256, size = (4, 3))
		action = np.random.randint(16)
		reward = np.random.random()
		terminal = False
		if np.random.random() < .05:
			terminal = True

		dataset1.add_experience(img, terminal, action, reward)
		dataset2.add_experience(img, terminal, action, reward)

		if dataset1.can_get_state() or dataset2.can_get_state():
			np.testing.assert_array_almost_equal(dataset1.get_state(img),
												dataset2.get_state(img))
	print "passed"

def main():
	trivial_tests()
	max_size_tests()

if __name__ == '__main__':
	main()
