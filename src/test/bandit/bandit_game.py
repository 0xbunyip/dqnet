import numpy as np

class BanditGame():
	"""docstring for GridGame"""

	MAX_MEAN = 4
	MAX_VAR = 2

	def __init__(self, height, width, one_state, rng):
		self.one_state = one_state
		self.rng = rng
		self.action_set = range(height * width)
		self.mean = rng.randint(-BanditGame.MAX_MEAN, BanditGame.MAX_MEAN + 1
								, size = (height, width))
		self.grid = None
		self.height = height
		self.width = width
		self.translate = BanditGame.MAX_MEAN + BanditGame.MAX_VAR
		self.reset_game()
		self.acted = False
		print self.mean

	def game_info(self):
		return "Bandit game (%dx%d), mean = %d, var = %d\n%s" % \
			(self.height, self.width, BanditGame.MAX_MEAN, BanditGame.MAX_VAR
			, self.reset_game())

	def getScreenDims(self):
		return (self.width, self.height)
		
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
		if self.one_state:
			np.copyto(dst = obs, src = np.zeros((self.height, self.width)
						, dtype = np.uint8))
		else:
			np.copyto(dst = obs, src = self.grid)
		
	def reset_game(self):
		self.acted = False
		return self._reset_game_full()

	def _normalize_grid(self):
		self.grid += self.translate
		self.grid = np.uint8(self.grid)

	def _reset_game_grid(self):
		self.grid = self.mean + self.rng.randint(-BanditGame.MAX_VAR
					, BanditGame.MAX_VAR + 1, size = (self.height, self.width))
		self._normalize_grid()
		return "Reset game grid (no reset mean)"

	def _reset_game_full(self):
		self.mean = self.rng.randint(-BanditGame.MAX_MEAN
					, BanditGame.MAX_MEAN + 1, size = (self.height, self.width))
		self._reset_game_grid()
		return "Reset game full (both mean and grid)"

	def game_over(self):
		return self.acted

	def lives(self):
		return 1