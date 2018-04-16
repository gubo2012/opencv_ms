#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 09:49:29 2018

@author: gubo
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


# Load a color image
img = cv2.imread("Images/starfish.png")

#cv2.imshow('original', img)
plt.figure(1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# Apply some blurring to reduce noise

# h is the Parameter regulating filter strength for luminance component. 
# Bigger h value perfectly removes noise but also removes image details, 
# smaller h value preserves details but also preserves some noise

# Hint: I recommend using larger h and hColor values than typical to remove noise at the
# expense of losing image details

# Experiment with setting h and hColor to a suitable value.

# Exercise: Insert code here to set values for h and hColor. 
# Hint: You'll find answers at the bottom of the lab. 
hColor = h = 20
    
# Default values
templateWindowSize = 7
searchWindowSize = 21
    
blur = cv2.fastNlMeansDenoisingColored(img, None,h,hColor,templateWindowSize,searchWindowSize)
plt.figure(2)    
plt.imshow(cv2.cvtColor(blur, cv2.COLOR_BGR2RGB))


hColor = h = 40
    
blur2 = cv2.fastNlMeansDenoisingColored(img, None,h,hColor,templateWindowSize,searchWindowSize)
plt.figure(3)    
plt.imshow(cv2.cvtColor(blur2, cv2.COLOR_BGR2RGB))


# Apply a morphological gradient (dilate the image, erode the image, and take the difference

elKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13,13))
elKernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))

# YOUR CODE HERE
# Exercise: Use openCV's morphologyEx to generate a gradient using the kernel above
# Hint: You'll find answers at the bottom of the lab. 

# END YOUR CODE HERE
gradient = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, elKernel)
#gradient2 = cv2.morphologyEx(blur2, cv2.MORPH_GRADIENT, elKernel)
gradient2 = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, elKernel2)

plt.figure(4)
plt.imshow(cv2.cvtColor(gradient, cv2.COLOR_BGR2RGB))
plt.figure(5)
plt.imshow(cv2.cvtColor(gradient2, cv2.COLOR_BGR2RGB))

