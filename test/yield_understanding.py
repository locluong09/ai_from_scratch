import numpy as np
for i in range(10):
	print(i)


def createGenerator():
	#this function aims to understand the yield in python
	x = np.arange(10)
	for i in x:
		yield np.power(i,3)

my_generator = createGenerator()
print(my_generator)

for i in my_generator:
	print(i)