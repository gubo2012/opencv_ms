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

# Apply a morphological gradient (dilate the image, erode the image, and take the difference

elKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13,13))

# YOUR CODE HERE
# Exercise: Use openCV's morphologyEx to generate a gradient using the kernel above
# Hint: You'll find answers at the bottom of the lab. 

# END YOUR CODE HERE
gradient = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, elKernel)


plt.figure(3)
plt.imshow(cv2.cvtColor(gradient, cv2.COLOR_BGR2RGB))

# Apply Otsu's method - or you can adjust the level at which thresholding occurs
# and see what the effect of this is

# Convert gradient to grayscale
gradient = cv2.cvtColor(gradient, cv2.COLOR_BGR2GRAY)


# YOUR CODE HERE
# Exercise: Generate a matrix called otsu using OpenCV's threshold() function.  Use
# Otsu's method.
# Hint: You'll find answers at the bottom of the lab. 

# END YOUR CODE HERE
otsu = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
       
# Apply a closing operation - we're using a large kernel here. By all means adjust the size of this kernel
# and observe the effects
closingKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (33,33))
close = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, closingKernel)

# Erode smaller artefacts out of the image - play with iterations to see how it works
    
# YOUR CODE HERE
# Exercise: Generate a matrix called eroded using cv2.erode() function over the 'close' matrix.
# Experiment until your output image is similar to the image below
# Hint: You'll find answers at the bottom of the lab. 

# END YOUR CODE HERE
eroded = cv2.erode(close, None, iterations=6)

plt.figure(4)
plt.imshow(eroded, cmap='gray')

# find contours
# this 4 lines of codes are not necessary
#eroded_copy = eroded.copy()
#p = int(img.shape[1] * 0.05)
#eroded[:, 0:p] = 0
#eroded[:, img.shape[1] - p:] = 0

# Exercise: Find the contours - just external contours to keep post-processing simple
# Hint: You'll find answers at the bottom of the lab. 
(cnting, contours, _) = cv2.findContours(eroded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_copy = contours.copy()

# Sort the candidates by size, and just keep the largest one
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

# Lets create two images, initially all zeros (i.e. black)
# One image will be filled with 'Blue' wherever we think there's some starfish
# The other image will be filled with 'Green' whereever we think there's not some starfish
h, w, num_c = img.shape
segmask = np.zeros((h, w, num_c), np.uint8) # background mask
stencil = np.zeros((h, w, num_c), np.uint8) # foreground stencil

# I know we've only one contour, but - in general - we'd expect to have more contours to deal with
for c in contours:
    # Fill in the starfish shape into segmask
    cv2.drawContours(segmask, [c], 0, (255, 0, 0), -1) # blue
    cv2.drawContours(segmask, [c], 0, (0, 0, 255), -1) 
    # Lets fill in the starfish shape into stencil as well
    # and then re-arrange the colors using numpy
    cv2.drawContours(stencil, [c], 0, (255, 0, 0), -1)
    stencil[np.where((stencil==[0,0,0]).all(axis=2))] = [0, 255, 0] 
    stencil[np.where((stencil==[255,0,0]).all(axis=2))] = [0, 0, 0]  

# Now, lets create a mask image by bitwise ORring segmask and stencil together
mask = cv2.bitwise_or(stencil, segmask)

plt.figure(5)
plt.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))

# Exercise: Blend the original image 'img' and our mask 'mask'
# in any way you see fit, and store it in a variable called output
output = cv2.bitwise_or(mask, img)
plt.figure(6)
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
#plt.close('all')

#plt.figure(7)
#plt.imshow(cv2.cvtColor(stencil, cv2.COLOR_BGR2RGB))
#plt.figure(8)
#plt.imshow(cv2.cvtColor(segmask, cv2.COLOR_BGR2RGB))