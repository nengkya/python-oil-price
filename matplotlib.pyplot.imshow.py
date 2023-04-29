import numpy as matrix
import numpy, matplotlib.pyplot as plt

'''
create matrix [1 0] x 10
[1 0 1 0 1 0 1 0 1 0],
'''
array1 =  numpy.array([
	[1, 0] * 10,
	[0, 1] * 10
] * 10)

print(array1)
plt.imshow(array1,origin="upper")
plt.show()
