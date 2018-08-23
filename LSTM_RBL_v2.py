#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 21:01:43 2018

@author: Nathaniel, Zikai Wei
"""
#import sys
#import os
#from glob import glob

#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt


import torch
#import torch.autograd as autograd
#import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.utils.data import DataLoader
#from torch.utils.data import Dataset, DataLoader
#from torch.optim import lr_scheduler


from models.model_lstmfc3net import LSTMFC3Net
from dataloaders.dataloader_bardatasetrw import BarDatasetRW


'''Import data and visulize it
'''
filename = '/home/weizikai/Documents/Python/R_breaker/RBL8.csv'


    
train_dataset= BarDatasetRW(filename, rws = 10, train_phase = True, train_ratio = 0.66)
val_dataset= BarDatasetRW(filename, rws = 10, train_phase = False, train_ratio = 0.66)

dataloader_train = DataLoader(train_dataset, batch_size = 256, shuffle = True, num_workers = 2, drop_last=True)
dataloader_test = DataLoader(val_dataset, batch_size = 256, shuffle = False, num_workers = 2, drop_last=True)


model = LSTMFC3Net()    
model = model.cuda()


optimizer = optim.Adam(model.parameters(),lr=0.05)
is_cuda = torch.cuda.is_available()


def fit(epoch, model, data_loader, phase = 'training', volatile = False):
    if phase == 'training':
        model.train()
    if phase == 'validation':
        model.eval()
        #volatile = True
    running_loss = 0.0
    running_correct = 0
    for batch_idx, batch in enumerate(data_loader):
        bars, target = batch['bars'], batch['labels']
        #if is_cuda:
        bars, target = bars.cuda(), target.cuda()
        bars, target = Variable(bars), Variable(target).long()
        if phase == 'training':
            optimizer.zero_grad()
        output = model(bars)
        loss = F.nll_loss(output, target)
        running_loss += F.nll_loss(output, target, size_average = False).data[0]
        preds = output.data.max(dim=1, keepdim = True)[1]
        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()
        if phase == 'training':
            loss.backward()
            optimizer.step()
    loss = running_loss / len(data_loader.dataset)
    accuracy = 100. * running_correct/ len(data_loader.dataset)
    print(f'{phase} loss is {loss:{5.6}} and {phase} accuracy is {running_correct}/{len(data_loader.dataset)}{accuracy:{10}.{4}}') 
    return loss, accuracy

train_losses, train_accuracy = [],[]
val_losses, val_accuracy = [],[]

for epoch in range(1,2):
    epoch_loss, epoch_accuracy = fit(epoch,model,dataloader_train, phase = 'training')
    val_epoch_loss, val_epoch_accuracy = fit(epoch,model,dataloader_test, phase = 'validation')
    train_losses.append(epoch_loss) 
    train_accuracy.append(epoch_accuracy)
    val_losses.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)


