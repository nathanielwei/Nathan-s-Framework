#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 15:51:16 2018

@author: weizikai
"""



'''
model_lstmfc3net.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class LSTMFC3Net(nn.Module): 

    def __init__(self, input_size = 5, batch = 256, hidden_size = 256, num_layers = 1, num_directions = 1):
        super().__init__()
        
        '''parameters for lstm: input_size = 5 # 5 for 5 features, batch = 256
                             hidden_size = 256, num_layers = 1, num_directions = 1 '''
        self.lstm1 = nn.LSTM(input_size,batch,num_layers, batch_first= True)  
        self.h0 = Variable(torch.randn(num_layers*num_directions, batch, hidden_size * num_directions).cuda())
        self.c0 = Variable(torch.randn(num_layers*num_directions, batch, hidden_size * num_directions).cuda())
        self.fc1 = nn.Linear(hidden_size*10, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)
        
    
    def forward(self, x, batch = 256):   
        x , hnn = self.lstm1(x, (self.h0, self.c0))
        x = x.contiguous().view(batch, -1)
        x = F.relu(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x) # dropout
        x = F.relu(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=0)