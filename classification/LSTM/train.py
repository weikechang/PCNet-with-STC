# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 15:24:34 2022

@author: mumu
"""
from torch import no_grad
import torch.nn as nn
import torch
import numpy as np
from LSTM import LSTM
from six.moves import xrange
from torch.optim import lr_scheduler
from numpy import arange
from torch.optim import Adam
import matplotlib.pyplot as plt

seed = 3
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda") 

############################################
#train net work
def train(model, device, train_data, train_label,batch_size, optimizer, scheduler, epoch, epochs):

    model.train()
    losses = 0
    train_size=len(train_data)
    loss_value = torch.zeros((int(train_size // batch_size)))
    for step in xrange(train_size // batch_size):
        offset = (step * batch_size) % (train_size)
        data = train_data[offset:(offset + batch_size)]
        target = train_label[offset:(offset + batch_size)]
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output,_,_,_ = model(data)
        loss_fn = nn.L1Loss(reduction='sum')
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        batch_loss = loss.item()
        losses += batch_loss
        batches = int(len(train_data) / len(data))
        loss_value[step]=batch_loss
# =============================================================================
#         print('Train: Epoch {e}/{e_max}, Batch {b}/{b_max}, Loss: {l:.5f}'.
#               format(e=epoch, e_max=epochs, b=step+1, b_max=batches, l=batch_loss))
# =============================================================================
    average = losses/(train_size // batch_size)
    print('Train: Epoch {e}/{e_max}, Loss: {l:.5f}'.format(e=epoch, e_max=epochs, l=average))
    return average

#test the trained model
def test(model, device, test_data, test_label, batch_size, epoch, epochs):
    model.eval()
    losses = 0 
    test_size=len(test_data)
    with no_grad():
        for step in xrange(test_size // batch_size):
            offset = (step * batch_size) % (test_size)
            data = test_data[offset:(offset + batch_size)]
            target = test_label[offset:(offset + batch_size)]
            data, target = data.to(device), target.to(device)
            output,_,_,_ = model(data)
            loss_fn = nn.L1Loss(reduction='sum')
            losses += loss_fn(output, target).item()
    average = losses/(test_size // batch_size)
    print('Test: Epoch {e}/{e_max},Loss:{average:.5f}'.
          format(e=epoch, e_max=epochs, average=average))
    return average


def main():
    # parameters
    batch_size = 50
    epochs = 600
    learning_rate = 1*1e-2
    input_len = 1
    num = 3 # length of label, you can change according to what you set in creat_train_data.py
    
    #load dataset
    A = np.load('../data/tactrain6_3.npz') # load data created form creat_train_data.py
    train_data = A['train_data']
    val_data = A['test_data']
    
    train_label = A['train_label']
    val_label = A['test_label']
    
    data_size = train_data.shape[1]
    label_size = train_label.shape[1]

    
    print (train_data.shape,train_label.shape)
    print (epochs,batch_size,learning_rate)
    
    train_data = train_data.reshape(-1, data_size, 1).astype('float32')    
    train_label = train_label.reshape(-1, label_size).astype('float32')
    
    val_data = val_data.reshape(-1, data_size, 1).astype('float32')
    val_label = val_label.reshape(-1, label_size).astype('float32')
    print(train_data.shape,val_data.shape)

    #data for torch

    train_data = torch.from_numpy(train_data)
    train_label = torch.from_numpy(train_label)
    
    val_data = torch.from_numpy(val_data)
    val_label = torch.from_numpy(val_label)
    

    

    # computing device
    model = LSTM(input_len = input_len,
              hidden = 64, # the number of cells in the lstm, you can change it to find a better one.
              layers = 1,
              num = num)

    model = model.to(device)

    # optimizer
    optimizer = Adam(params=model.parameters(), lr=learning_rate, weight_decay=1e-3)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    
    print (model)
    LOSS_train_plot = []
    LOSS_test_plot = []
    ax=[]
    plt.ion()
    for epoch in arange(start=1, stop=(epochs + 1), step=1, dtype=int):
        LOSS=train(model=model, device=device, train_data = train_data, train_label = train_label, batch_size=batch_size, optimizer=optimizer, scheduler=scheduler, epoch=epoch, epochs=epochs)
        scheduler.step()
        print(epoch, 'lr={:.6f}'.format(optimizer.param_groups[0]['lr']))
        LOSSt=test(model=model, device=device, test_data = val_data, test_label = val_label, batch_size=batch_size, epoch=epoch, epochs=epochs)
        LOSS=np.array(LOSS)
        LOSSt=np.array(LOSSt)
        LOSS_train_plot.append(LOSS)
        LOSS_test_plot.append(LOSSt)
        ax.append(epoch)
        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.plot(ax,LOSS_train_plot,'r')
        plt.plot(ax,LOSS_test_plot,'b')
        
        plt.pause(0.05)

    plt.ioff()
    plt.show()
    torch.save(model.state_dict(), 'D:/Plant_classification/PCA_SVM11/11/LSTM/model_save/1LSTMmodel_tlstage64.pth')

    
if __name__ == '__main__':
    main()
