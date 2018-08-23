#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 16:13:15 2018

@author: weizikai
"""

'''
dataloader_bardatasetrw.py
'''

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset



class BarDatasetRW(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, rws, train_phase, train_ratio, transform = None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        ct = np.asarray(self.data_frame["close"][1:len(self.data_frame)])
        c_t_p = np.asarray(self.data_frame["close"][0:len(self.data_frame)-1])
        c = ct-c_t_p  # 1: down  # 2: up
        d = np.zeros(len(c))
        d[c>=3] = 2  # 1: down  # 2: up
        d[c<=-3] = 1 

        self.data_frame["none"][1:len(self.data_frame)] = d.copy()
        #.reshape(len(d)) 
        
        self.rws = rws
        self.train_phase = train_phase
        self.train_ratio = train_ratio
        
        # self.list = range(0,len(data)-rws+1) # the last observation cannot be used to predict or validate
        last_idx = len(self.data_frame)-rws+1
        last_train_idx = int(last_idx*train_ratio)
        self.list_train = range(0, last_train_idx)
        self.list_dev = range(last_train_idx, last_idx) # the last observation cannot be used to predict or validate
        
        self.transform = transform

        

    def __len__(self):
        if self.train_phase == True:
            data_len = len(self.list_train)
        elif self.train_phase == False:
            data_len = len(self.list_dev)
        return data_len
        #len(self.data_frame)- self.rws + 1

    def __getitem__(self, idx):
        if self.train_phase == True:
            self.list = self.list_train
        elif self.train_phase == False:
            self.list = self.list_dev
        idx_bgn = self.list[idx]
        idx_end = idx_bgn+ self.rws

        x = np.concatenate((self.data_frame["open"][idx_bgn:idx_end].values.reshape(self.rws,1), 
                self.data_frame["high"][idx_bgn:idx_end].values.reshape(self.rws,1),
                self.data_frame["low"][idx_bgn:idx_end].values.reshape(self.rws,1),
                self.data_frame["close"][idx_bgn:idx_end].values.reshape(self.rws,1),
                self.data_frame["volume"][idx_bgn:idx_end].values.reshape(self.rws,1)),
                axis=1)
        #instance_x = torch.from_numpy(x).type(torch.FloatTensor).view(self.rws,1,5)
        

        instance_x = torch.from_numpy(x).type(torch.FloatTensor).view(self.rws,5)

        instance_y =  self.data_frame["none"][self.rws + idx]
        if self.transform:
            instance_x = self.transform(instance_x)
            
        sample = {'bars': instance_x, 'labels': instance_y}


        return sample