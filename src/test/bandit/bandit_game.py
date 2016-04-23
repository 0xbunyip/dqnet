import numpy as np

class BanditGame():
	"""docstring for GridGame"""

	MAX_MEAN = 3
	MAX_VAR = 2

	def __init__(self, height, width, rng):
		self.rng = rng
		self.action_set = range(height * width)
		self.mean = np.random.randint(-BanditGame.MAX_MEAN, BanditGame.MAX_MEAN + 1, size = (height, width))
		self.grid = None
		self.height = height
		self.width = width
		self.translate = BanditGame.MAX_MEAN + BanditGame.MAX_VAR
		self.reset_game()
		self.acted = False
		print self.mean
		
	def getMinimalActionSet(self):
		return self.action_set

	def act(self, a):
		if self.game_over():
			return 0
		assert a < len(self.action_set)
		self.acted = True
		x = a // self.width
		y = a % self.width
		return self.grid[x, y] - self.translate

	def getScreenGrayscale(self, obs):
		np.copyto(dst = obs, src = self.grid)
		# np.copyto(dst = obs, src = np.zeros((2, 2), dtype = np.uint8))

	def reset_game(self):
		self.acted = False
		self._reset_game_grid()

	def _normalize_grid(self):
		self.grid += self.translate
		self.grid = np.uint8(self.grid)

	def _reset_game_grid(self):
		self.grid = self.mean + self.rng.randint(-BanditGame.MAX_VAR, BanditGame.MAX_VAR + 1, size = (self.height, self.width))
		self._normalize_grid()

	def _reset_game_full(self):
		self.mean = np.random.randint(-BanditGame.MAX_MEAN, BanditGame.MAX_MEAN + 1, size = (height, width))
		self._reset_game_grid()

	def game_over(self):
		return self.acted

	def lives(self):
		return 1