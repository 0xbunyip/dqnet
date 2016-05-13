import numpy as np
import unittest
import os.path, sys
import theano
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir))
from experience import Experience
from collections import deque

class TestExp(unittest.TestCase):
	def setUp(self):
		self.rng = np.random.RandomState(123)
		self.t_prob = 0.1
		self.n_act = 4
		self.m_rew = 10

	def rotate_array(self, a, exp):
		if exp.len < exp.replay_mem_size:
			return a
		return a.take(exp.top + np.arange(exp.len), axis = 0, mode = 'wrap')

	def test_max_length(self):
		height = 2
		width = 2
		channel = 2
		exp = Experience(10, height, width, channel, self.rng)
		dexp = deque(maxlen = 10)
		N = self.rng.randint(10, 1000)
		for i in xrange(N):
			obs = np.ones((height, width)).astype(np.uint8) * (i % 256)
			is_terminal = True if self.rng.rand() < self.t_prob else False
			action = np.uint8(self.rng.randint(self.n_act))
			reward = self.rng.randint(self.m_rew * 2) - self.rng.randint(self.m_rew)
			exp.add_experience(obs, is_terminal, action, reward)
			dexp.append((obs, is_terminal, action, reward))

		lexp = list(dexp)

		np.testing.assert_allclose(np.ones((2, 2)).astype(np.uint8) * (N - 1) % 256
			, exp.obs[exp.top - 1], rtol = 0.00001)

		lo, lt, la, lr = zip(*lexp)
		np.testing.assert_allclose(np.array(lo).astype(np.uint8)
			, self.rotate_array(exp.obs, exp), rtol = 0.00001)

		np.testing.assert_allclose(np.array(lt).reshape(-1, 1).astype(np.bool_)
			, self.rotate_array(exp.terminal, exp), rtol = 0.00001)

		np.testing.assert_allclose(np.array(la).reshape(-1, 1).astype(np.uint8)
			, self.rotate_array(exp.action, exp), rtol = 0.00001)

		np.testing.assert_allclose(np.array(lr).reshape(-1, 1).astype(np.float32)
			, self.rotate_array(exp.reward, exp), rtol = 0.00001)

	def test_get_state(self):
		height = 2
		width = 2
		channel = 4
		exp = Experience(10, height, width, channel, self.rng)
		dexp = deque(maxlen = 10)
		N = self.rng.randint(channel, 1000)
		for i in xrange(N):
			obs = self.rng.randint(0, 256, (height, width)).astype(np.uint8)
			is_terminal = True if self.rng.rand() < self.t_prob else False
			action = np.uint8(self.rng.randint(self.n_act))
			reward = self.rng.randint(self.m_rew * 2) - self.rng.randint(self.m_rew)
			exp.add_experience(obs, is_terminal, action, reward)
			dexp.append((obs, is_terminal, action, reward))
			if i >= channel and self.rng.rand() < 0.1:
				exp.get_state(obs)

		obs = self.rng.randint(0, 256, (height, width)).astype(np.uint8)
		dobs = [obs]
		for j in xrange(channel - 1):
			dobs.insert(0, dexp.pop()[0])

		np.testing.assert_allclose(np.array(dobs).astype(np.uint8)
			, exp.get_state(obs), rtol = 0.00001)

	def test_one_long_state(self):
		height = 2
		width = 2
		N = self.rng.randint(4, 1000)
		channel = N
		exp = Experience(N, height, width, channel, self.rng)

		for i in xrange(N - 1):
			obs = np.ones((height, width)).astype(np.uint8) * (i % 256)
			is_terminal = False
			action = np.uint8(self.rng.randint(self.n_act))
			reward = self.rng.randint(self.m_rew * 2) - self.rng.randint(self.m_rew)
			exp.add_experience(obs, is_terminal, action, reward)
			self.assertEqual(i + 1 >= N - 1, exp.can_get_state())

		obs = np.ones((height, width)).astype(np.uint8) * ((N - 1) % 256)
		fobs = np.ones((channel, height, width)).astype(np.uint8)
		for i in xrange(N):
			fobs[i, ...] *= i % 256
		np.testing.assert_allclose(fobs
			, exp.get_state(obs), rtol = 0.00001)

	def test_get_random_minibatch(self):
		height = 2
		width = 2
		channel = 4
		N = 100
		mbsize = 32
		exp = Experience(10, height, width, channel, self.rng)
		terminal = np.bool_([0, 1, 1, 0, 0, 0, 1, 0, 1, 0])

		for i in xrange(N):
			obs = np.ones((height, width)).astype(np.uint8) * (i % 10)
			is_terminal = terminal[i % 10]
			action = np.uint8(self.rng.randint(self.n_act))
			reward = self.rng.randint(self.m_rew * 2) - self.rng.randint(self.m_rew)
			exp.add_experience(obs, is_terminal, action, reward)

		states, action, reward, terminal = exp.get_random_minibatch(mbsize)
		fstates = np.ones((mbsize, channel + 1, height, width)).astype(np.uint8)
		for j in xrange(channel + 1):
			fstates[:, j, :, :] = j + 3

		np.testing.assert_allclose(fstates
			, states, rtol = 0.00001)

if __name__ == '__main__':
	unittest.main()