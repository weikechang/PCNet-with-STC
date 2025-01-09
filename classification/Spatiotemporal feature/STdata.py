# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 15:30:29 2020

@author: mumu
"""
import os
import numpy as np
def nor(x):
    return (x - np.min(x))/(np.max(x)-np.min(x))

path = ".../normal_MTS" #path of normal MTS
files= os.listdir(path) 
files.sort(key=lambda x: int(x.split('.')[0][0:3]))
TAC_n=[]
for file in files: 
    A = np.load(path+'/'+ file)
    da = A['hn']
    da_cutout=da.flatten()
    TAC_n.append(da_cutout)
    print(file)
TAC_n=np.array(TAC_n)

path1 = ".../normal_image" #path of normal image
files1 = os.listdir(path1) 
files1.sort(key=lambda x: int(x.split('.')[0][0:3]))
img_n=[]
for file1 in files1: 
    B = np.load(path1 +'/'+ file1)
    da1 = B['plant_fea']
    da1 = da1.flatten()
    img_n.append(da1)
    print(file1)
img_n = np.array(img_n)
print(np.max(img_n),np.min(img_n))
print(img_n.shape)

path2 =  ".../stress_MTS" #path of stress MTS
files2 = os.listdir(path2) 
files2.sort(key=lambda x: int(x.split('.')[0][0:3]))
TAC_n8=[]
for file2 in files2: 
    C = np.load(path2 +'/'+ file2)
    da2 = C['hn']
    da2 = da2.flatten()
    TAC_n8.append(da2)
    print(file2)
TAC_s = np.array(TAC_n8)

path3 = ".../stress_image" #path of stress image
files3 = os.listdir(path3) 
files3.sort(key=lambda x: int(x.split('.')[0][0:3]))
img_n8=[]
for file3 in files3: 
    D = np.load(path3 +'/'+ file3)
    da3 = D['plant_fea']
    da3 = da3.flatten()
    img_n8.append(da3)
    print(file3)
img_s = np.array(img_n8)


img = np.row_stack((img_n,img_s))
TAC = np.row_stack((TAC_n,TAC_s))

from sklearn.decomposition import PCA

n_label=np.ones((15,1))
s_label=np.zeros((9,1))
label = np.row_stack((n_label,s_label)).squeeze()
path = '.../PCnet-and-STC/classification/SVM/data/'
for i in range(3,25,2):
  pca1 = PCA(n_components=i)
  newimg = pca1.fit_transform(img)
  newimg = nor(newimg)
  for j in range(3,25,2):
    pca2 = PCA(n_components=j)
    newTAC = pca2.fit_transform(TAC)
    newTAC = nor(newTAC)
    newdata = np.column_stack((newimg,newTAC))
    if i < 10:
        if j < 10:
            np.savez(path+'spa'+'tem'+str(0)+str(i)+str(0)+str(j)+'.npz',data=newdata,label=label)
        if j >= 10:
            np.savez(path+'spa'+'tem'+str(0)+str(i)+str(j)+'.npz',data=newdata,label=label)
            
    if i >= 10:
        if j < 10:
            np.savez(path+'spa'+'tem'+str(i)+str(0)+str(j)+'.npz',data=newdata,label=label)
        if j >= 10:
            np.savez(path+'spa'+'tem'+str(i)+str(j)+'.npz',data=newdata,label=label)
