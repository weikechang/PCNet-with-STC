# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 10:28:47 2019

@author: mumu
"""
import torch
import numpy as np
device = torch.device("cuda")
import matplotlib.pyplot as plt

# =============================================================================
from LSTM import LSTM
# =============================================================================
def nor(x):
    return (x - np.min(x))/(np.max(x)-np.min(x))

def predict(data):
    state_dict = torch.load('.../LSTM/model_save/LSTMmodel_tlstage64.pth')
    model = LSTM(input_len = 1,
              hidden = 64,
              layers = 1,
              num = 3)
    #print(model)    
    model.load_state_dict(state_dict)
    model.to(device)

    data = data.to(device)
    model.eval()
    with torch.no_grad():
        output_v, x_out, hn, cn = model(data)
        #torch.cuda.empty_cache()
    return output_v, x_out, hn, cn
import os
path = '.../data/test/'
files = os.listdir(path)
len_seq = 6
num_out = 3
num = (24 - len_seq) / num_out
for file in files:
    A = np.load(path+file)
    TAC = A['TAC'].astype('float32')  
    HN = []
    OUTPUT = []
    CN = []
    for j in range(TAC.shape[0]):
        data = TAC[j,0:len_seq]/np.max(TAC[j]).astype('float32')  
        data = data.reshape(1,data.shape[0])
        H = []
        C= []
        output = data
        for i in range(int(num)):
            data1 = data.reshape(-1, data.shape[1], 1).astype('float32')  
            data1 = torch.from_numpy(data1)
            prediction,x_out,hn,_ = predict(data1)
            prediction = prediction.cpu().detach().numpy()
            hn = hn.cpu().detach().numpy().squeeze()
            x_out = x_out.cpu().detach().numpy().squeeze()
            H.append(hn)
            C.append(x_out)
            
            data = np.concatenate((data[:,prediction.shape[1]:data.shape[1]],prediction),axis = 1)
            output = np.concatenate((output,prediction),axis = 1)
        H = np.array(H)
        C = np.array(C)
        output = np.array(output.squeeze())
        HN.append(H)
        CN.append(C)
        OUTPUT.append(output)
    HN = np.array(HN)
    CN = np.array(CN)
    OUTPUT = np.array(OUTPUT)
    name = str(file.split('.')[0])
    #np.savez('.../hn/'+name+'.npz',hn= HN)

    print(file,OUTPUT.shape)


