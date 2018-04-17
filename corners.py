#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 14:21:44 2018

@author: gubo
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the martini image as grayscale.
img = cv2.imread("Images/martini.png")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)

corners = cv2.cornerHarris(gray, 2, 3, 0.04)

img_i = 0
img_i += 1
plt.figure(img_i)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

threshold = 0.001

img[corners>threshold * corners.max()]=[0, 0, 255]

img_i += 1
plt.figure(img_i)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))