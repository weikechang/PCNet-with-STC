# -*- coding: utf-8 -*-
"""
@author: mumu
"""
import numpy as np
import matplotlib.pyplot as plt
import copy
A = np.load('...../classification/Spatial Feature Extractor/data/1109middle.npz')
img = A['image']
m = 10000
orimg = img.mean(axis=0)
img1 = copy.copy(orimg)
img1[:,:,60:105] = 0
img1 = img1.transpose(2, 0, 1)
img1[img1<m] = 0
print(img1.shape)
position = np.where(img1!=0)
mask = np.zeros((img1.shape[0],img1.shape[1],img1.shape[2]))
mask[position] = 1
feature = img1[position]
print(feature.shape)
plt.figure()
plt.imshow(orimg.transpose(2, 0, 1).max(axis=2), cmap ="hot_r",vmin=0, vmax=30000)
plt.figure()
plt.imshow(img1.max(axis=2), cmap ="hot_r",vmin=0, vmax=30000)
for i in range(0,24,8):
    plt.figure()
    newimg = (img[i].transpose(2, 0, 1))*mask
    plt.imshow(newimg.max(axis=2),cmap ="hot_r",vmin=0, vmax=30000)
plt.show()