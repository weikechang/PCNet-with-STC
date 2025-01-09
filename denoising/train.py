# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 15:24:34 2022

@author: mumu
"""
from torch import no_grad
import torch.nn.functional as F
import torch
import numpy as np
#from arch_unet import UNet
from PCNet import MEM_CNN
from six.moves import xrange
from torch.optim import lr_scheduler
from numpy import arange
from torch.optim import Adam


seed = 3
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda") 

############################################
#train net work
def train(model, device, train_data, train_label,batch_size, optimizer, epoch, epochs):
    
    model.train()
    increase_ratio = 0.3
    losses = 0
    train_size=len(train_data)
    loss_value = torch.zeros((int(train_size // batch_size)))
    for step in xrange(train_size // batch_size):
        offset = (step * batch_size) % (train_size)
        data = train_data[offset:(offset + batch_size)]
        target = train_label[offset:(offset + batch_size)]
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        with torch.no_grad():
              target_denoised = model(target)
              res = model(data - target)
        
        output = model(data)
        Lambda = increase_ratio * epoch / epochs
        diff = output - target
        exp_diff = res - (output - target_denoised)
        
        loss1 = torch.mean(diff**2)
        loss2 = Lambda * torch.mean((exp_diff)**2)
        loss = loss1 + loss2

        loss.backward()
        optimizer.step()
        batch_loss = loss.item()
        losses += batch_loss
        batches = int(len(train_data) / len(data))
        loss_value[step]=batch_loss
        print('Train: Epoch {e}/{e_max}, Batch {b}/{b_max}, Loss: {l:.5f}'.
              format(e=epoch, e_max=epochs, b=step+1, b_max=batches, l=batch_loss))
    average = losses/(train_size // batch_size)
    print (average)
    return average

#test the trained model
def val(model, device, test_data, test_label, batch_size, epoch, epochs):
    model.eval()
    increase_ratio = 0.3
    losses = 0 
    test_size=len(test_data)
    with no_grad():
        for step in xrange(test_size // batch_size):
            offset = (step * batch_size) % (test_size)
            data = test_data[offset:(offset + batch_size)]
            target = test_label[offset:(offset + batch_size)]
            data, target = data.to(device), target.to(device)
            target_denoised = model(target)
            output = model(data)
            res = model(data - target)
            Lambda = increase_ratio * epoch / epochs
            diff = output - target
            exp_diff = res - (output - target_denoised)
        
            loss1 = torch.mean(diff**2)
            loss2 = Lambda * torch.mean((exp_diff)**2)
            loss = loss1 + loss2

            losses +=loss.item()
    average = losses/(test_size // batch_size)
    print('Test: Epoch {e}/{e_max}, Average Loss: {l:.5f}'.
          format(e=epoch, e_max=epochs, l=average))
    return average


def main():
    # parameters
    batch_size = 2
    epochs = 100
    learning_rate = 1*1e-3
    
    #load dataset
    A = np.load('.../train_data.npz')
    train_data = A['train_data'][0:3000].astype('float32')
    val_data = A['train_data'][3000:3200].astype('float32')
    
    train_label = A['test_data'][0:3000].astype('float32')
    val_label = A['test_data'][3000:3200].astype('float32')
    
    data1_size = train_data.shape[1]
    data2_size = train_data.shape[2]
    data3_size = train_data.shape[3]

    
    print (train_data.shape)
    print (epochs,batch_size,learning_rate)
    
    train_data = train_data.reshape(-1, 1, data1_size, data2_size, data3_size).astype('float32')    
    train_label = train_label.reshape(-1, 1, data1_size, data2_size, data3_size ).astype('float32')  
    
    val_data = val_data.reshape(-1, 1, val_data.shape[1], val_data.shape[2], val_data.shape[3]).astype('float32')
    val_label = val_label.reshape(-1, 1, val_label.shape[1], val_label.shape[2], val_data.shape[3]).astype('float32')

    #data for torch

    train_data = torch.from_numpy(train_data)
    train_label = torch.from_numpy(train_label)
    
    val_data = torch.from_numpy(val_data)
    val_label = torch.from_numpy(val_label)
    

    

    # computing device
    model = MEM_CNN()
    
    state_dict = torch.load('/home/feng/CWK/PGCNN/self-supervised-noerr/PLANTPETdenoising/2ge/0/model_save/60model_tlstage.pth',map_location='cuda')
    model_dict = model.state_dict()
    new_state_dict = {k:v for k,v in state_dict.items() if k in model_dict}	
    model_dict.update(new_state_dict)	
    model.load_state_dict(model_dict)

    model = torch.nn.DataParallel(model, device_ids=[0,1])
    model = model.to(device)

    # optimizer
    optimizer = Adam(params=model.parameters(), lr=learning_rate, weight_decay=1e-3)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    print (model)
    for epoch in arange(start=1, stop=(epochs + 1), step=1, dtype=int):
        train(model=model, device=device, train_data = train_data,  train_label = train_label, batch_size=batch_size, optimizer=optimizer, epoch=epoch, epochs=epochs)
        val(model=model, device=device, test_data = val_data, test_label=val_label, batch_size=batch_size, epoch=epoch, epochs=epochs)
        scheduler.step()
        print(epoch, 'lr={:.6f}'.format(optimizer.param_groups[0]['lr']))
# =============================================================================
        if (epoch/1) in range(1,10001):
            #torch.save(model.module, '/home/feng/CWK/PGCNN/self-supervised-noerr/denoise/model_save/'+str(epoch)+'model_klstage.pkl') # for test the trained model
            torch.save(model.module.state_dict(), '/home/feng/CWK/PGCNN/self-supervised-noerr/PLANTPETdenoising/2ge/0.3/model_save/'+str(epoch)+'model_tlstage.pth')  # for the sake of Transfer learning
# =============================================================================



    
if __name__ == '__main__':
    main()
