import numpy as np

class GradientDescent():
	def __init__(self, learning_rate = 0.01):
		self.learning_rate = learning_rate
		self.weight_update = None

	def get_update(self, weight, gradient):
		if self.weight_update is None:
			self.weight_update = np.zeros(np.shape(weight))

		self.weight_update = gradient
		return weight - self.learning_rate*self.weight_update

