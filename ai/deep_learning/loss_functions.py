import numpy as np
import math

class SquareLoss():

	def loss(self, y, y_predict):
		length = len(y)
		return 1/(2*length)*(np.power(y - y_predict, 2))

	def derivative(self, y, y_predict):
		length = len(y)
		return 1/(length)*(y_predict - y)

	def RMSE(self, y, y_predict):
		return math.sqrt(np.mean(np.power(y - y_predict, 2)))


