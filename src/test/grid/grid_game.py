import numpy as np

class GridGame():
	"""docstring for GridGame"""

	MAX_MEAN = 4
	MAX_VAR = 2

	DX = [-1, 0, 1, 0]
	DY = [0, 1, 0, -1]
	def __init__(self, height, width, rng):
		self.rng = rng
		self.action_set = [0, 1, 2, 3] # Up, right, down, left
		self.mean = rng.randint(-GridGame.MAX_MEAN, GridGame.MAX_MEAN + 1, size = (height, width))
		self.grid = np.zeros((height, width), dtype = np.uint8)
		self.height = height
		self.width = width
		self.start = (0, 0)
		self.end = (0, 0)
		self.translate = GridGame.MAX_MEAN + GridGame.MAX_VAR
		self.reset_game()
		
	def getMinimalActionSet(self):
		return self.action_set

	def act(self, a):
		if self.game_over():
			return 0
		next_start = (self.start[0] + GridGame.DX[a], self.start[1] + GridGame.DY[a])
		if next_start[0] < 0 or next_start[1] < 0:
			return -1
		if next_start[0] >= self.height or next_start[1] >= self.width:
			return -1
		reward = self.grid[next_start] - self.translate
		self.grid[self.start] = -1 + self.translate
		self.grid[next_start] = 0 + self.translate
		self.start = next_start
		return reward

	def getScreenGrayscale(self, obs):
		np.copyto(dst = obs, src = self.grid)

	def reset_game(self):
		self._reset_game_full()

	def _normalize_grid(self):
		self.grid[self.grid == 0] = 1
		self.grid[self.start] = 0
		self.grid[self.end] = 9
		self.grid += self.translate
		self.grid = np.uint8(self.grid)

	def _reset_game_fixed(self):
		self.start = (self.height - 1, 0)
		self.end = (0, self.width - 1)
		self.grid = np.array([[-2, -1, -3, 9], [-1, 2, -1, 2], [3, -2, -2, 1], [0, -1, 2, -3]])
		self._normalize_grid()		

	def _reset_game_grid(self):
		self.start = (0, self.width - 1)
		self.end = (self.height - 1, 0)
		self.grid = self.mean + self.rng.randint(-GridGame.MAX_VAR, GridGame.MAX_VAR + 1, size = (self.height, self.width))
		self._normalize_grid()

	def _reset_game_full(self):
		self.start = (self.rng.randint(self.height), self.rng.randint(self.width))
		self.end = (self.rng.randint(self.height), self.rng.randint(self.width))
		while self.start == self.end:
			self.end = (self.rng.randint(self.height), self.rng.randint(self.width))
		
		self.mean = self.rng.randint(-GridGame.MAX_MEAN, GridGame.MAX_MEAN + 1, size = (self.height, self.width))
		self.grid = self.mean + self.rng.randint(-GridGame.MAX_VAR, GridGame.MAX_VAR + 1, size = (self.height, self.width))
		self._normalize_grid()

	def game_over(self):
		return self.start == self.end	

	def lives(self):
		return 1