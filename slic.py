#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 11:24:51 2018

@author: gubo
"""

from skimage import segmentation, color
from skimage.io import imread
from skimage.io import imsave

# Load an image
img = imread("Images/tomatoes.png")

# Exercise: Vary n_segments to find a reasonable value for it for this image
# Important Note: You are manually selecting features and feature properties 
# (e.g. size of super-pixels) here

numSegments = 800

compactFactor = 20

img_segments = segmentation.slic(img, compactness=compactFactor, n_segments=numSegments)
superpixels = color.label2rgb(img_segments, img, kind='avg')

plt.imshow(superpixels)