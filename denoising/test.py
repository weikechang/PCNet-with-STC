# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 10:28:47 2019

@author: mumu
"""
import torch
import numpy as np
import math
device = torch.device("cuda:1")
import matplotlib.pyplot as plt
from PCNet import MEM_CNN
import copy  
############################################################
    
def tensor2uint(img):
    img = img.data.squeeze().float().cpu().numpy()
    img[img<0] = 0
    return img


def prediect(data):
    state_dict = torch.load('/home/feng/CWK/PGCNN/self-supervised-noerr/PLANTPETdenoising/2ge/0.3/60model_save/60model_tlstage.pth',map_location='cuda:1')
    model = MEM_CNN()
    #print(model)    
    model.load_state_dict(state_dict)
    model.to(device)

    data = data.to(device)
    model.eval()
    with torch.no_grad():
        output_v = model(data)
        #torch.cuda.empty_cache()
    return output_v


import os
path = '/home/feng/CWK/植物数据去噪/data/'
files = os.listdir(path)
for file in files:
  path1 = path + file
  files1 = os.listdir(path1)
  name = str(file)
  print(name)
  if os.path.exists('/home/feng/CWK/植物数据去噪/PCnet/去噪数据/'+name) == False:
    os.mkdir('/home/feng/CWK/植物数据去噪/PCnet/去噪数据/'+name)
  for file1 in files1:
    A = np.load(path1 +'/'+ file1)
    image = A['data']
    data2_size = image.shape[0]
    data3_size = image.shape[2]
    output = np.zeros((data2_size, 1, data3_size)).astype('float32')
    k = 32
    print(file1)
    for i in range(int(image.shape[1]/k)):
        train_data1 = image[:,i*k:(i+1)*k,:]
        train_data1 = train_data1.reshape(-1, 1, data2_size, k, data3_size).astype('float32')
        train_data1 = torch.from_numpy(train_data1)
        net_output = prediect(train_data1)
        net_output = tensor2uint(net_output)
        output = np.concatenate((output,net_output),axis=1)
    output = output[:,1:output.shape[1],:].astype('float32') 
    dic = file1.split('.')[0]
    print(dic)
    np.savez_compressed('/home/feng/CWK/植物数据去噪/PCnet/去噪数据/'+name+'/'+str(dic)+'.npz',image = output)