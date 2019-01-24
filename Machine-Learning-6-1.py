# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 10:51:36 2019

@author: Yuan
"""

import matplotlib.pyplot as plt
from PIL import Image

img_color = Image.open("img/6/mnist_color.png")
img_gray = img_color.convert('1')

fig, (ax1, ax2) = plt.subplots(1,2)

ax1.axis("off")
ax2.axis("off")
ax1.set_title('colored')
ax1.imshow(img_color)
ax2.set_title('gray')
ax2.imshow(img_gray)
