'''
In image processing, a kernel, convolution matrix, or mask
is a small matrix used for blurring, sharpening, embossing,
edge detection, and more.

This is accomplished by doing a convolution
between the kernel and an image.

Or more simply, when each pixel in the output image
is a function of the nearby pixels (including itself)
in the input image, the kernel is that function.
'''

import os
import numpy as np

from matplotlib.image import imread
import matplotlib.pyplot as plt


os.system('t')

animal = imread('/home/haga/Desktop/animal.jpeg')
#animal picture size in 180 wide, 308 long
#3rd color dimension.
#black and white coolor (180, 308)
print(animal.shape)

#print(animal[0, 0, :3])

