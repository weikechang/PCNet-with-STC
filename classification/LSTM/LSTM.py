# -*- coding: utf-8 -*-
"""
Created on Tue May 31 09:28:19 2022

@author: mumu
"""

import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self,input_len, hidden, layers, num):
        super(LSTM, self).__init__()
        
        self.lstm = nn.LSTM(input_size = input_len, hidden_size = hidden, num_layers = layers, batch_first = True)
        self.linear = nn.Linear(hidden,num, bias = False)
        
    def forward(self, x):
        x_out, (hn, cn) = self.lstm(x)
        x1 = x_out[:,-1,:]
        y = self.linear(x1)
        
        return y, x_out, hn, cn
    
# =============================================================================
# if __name__ == '__main__':
#     vit = GRU(input_len=1,
#               hidden=20,
#               layers=1,
#               num=6)
#     a = torch.randn(5, 12, 1)
#     print(sum(p.numel() for p in vit.parameters()))
#     print(vit(a).shape)
# =============================================================================
