#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 10:05:18 2018

@author: gubo
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image
img = cv2.imread("Images/tomatoes.png")

blur = cv2.GaussianBlur(img, (21,21), 0)

# Convert to grayscale so we can threshold it
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

plt.figure(1)
plt.imshow(gray, cmap='gray')


# Exercise: So, create a matrix thresh which is thresholded by OpenCV's threshold() operation.
# HINT: I did not using OTSU here as I got better results from 
# weighting the threshold high. You can experiment here with thresholding to 
# improve on the final segmentation.
#
# Use the method cv2.threshold() which returns two values. The second value is the
# value you are interested in, you can ignore the first for now...
#
# Store the value in a variable called threshold
_, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

plt.figure(2)
plt.imshow(thresh, cmap='gray')


#Prepare the marker image

#  Close the image a little to fill in a few small holes in it
closingKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

# Create a matrix closed that is generated from thresh by a closing
# operation using the kernel above.
img_i = 2
#img_i += 1
#plt.figure(img_i)
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, closingKernel)
#plt.imshow(closed, cmap='gray')

# Use just enough dilate to get some clearly identifiable background
dilationKernel = np.ones((3,3), np.uint8)

# create a matrix 'bg' from OpenCV's dilate() function
# using the dilation kernel above
#img_i += 1
#plt.figure(img_i)
bg = cv2.dilate(closed, dilationKernel, iterations=3)
#plt.imshow(bg, cmap='gray')

# Now use a distance transform to extract is clearly foreground

# Create a matrix 'dist_transform' using OpenCV's distanceTransform
# method on the 'closed' matrix.
dist_transform = cv2.distanceTransform(closed, cv2.DIST_L2, 5)
img_i += 1
plt.figure(img_i)
plt.imshow(dist_transform, cmap='gray')

# Threshold the distance transformation
ret, fg = cv2.threshold(dist_transform,0.7*dist_transform.max(), 255, 0)

# Now find the unknown region by subtracting one from the other
fg = np.uint8(fg)

#img_i += 1
#plt.figure(img_i)
#plt.imshow(fg, cmap='gray')
unknown = cv2.subtract(bg, fg)

img_i += 1
plt.figure(img_i)
plt.imshow(unknown, cmap='gray')

# Marker labelling
ret, marker = cv2.connectedComponents(fg)

# Add one to all labels so that bg is not  0, but 1
marker = marker+1

# Now, mark the region of unknown with 0
marker[unknown==255] = 0;

img_i += 1
plt.figure(img_i)
plt.imshow(marker)


# Now marker is ready.  It is time for last step
cv2.watershed(img, marker)

# Create a new empty image with the same shape
# as the original image.
h, w, num_c = img.shape
seg = np.zeros((h, w, num_c), np.uint8)

# Watershed has replaced the pixel
# values in marker with integers representing
# the segments it has found in the original
# image.
# Color in these segments
# 
maxMarker = np.max(marker)
minMarker = np.min(marker)

colorMap =  [ \
             [0,0,0], \
             [255,255,255], \
             [127,0,0], \
             [0,0,255], \
             [0,255,0], \
             [255,0,0], \
             [255,255,0], \
             [255,0,255], \
             [0,255,255] \
            ]

for region in range(minMarker, maxMarker+1):
    seg[marker==region] = colorMap[region+1]

img_i += 1
plt.figure(img_i)    
plt.imshow(cv2.cvtColor(seg, cv2.COLOR_BGR2RGB))