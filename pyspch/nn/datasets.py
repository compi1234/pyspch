#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.utils.data import Dataset

### DataSet ###

class SpchDataset(Dataset):
    
    def __init__(self, corpus, input, target):
        
        # input = (time_dim, feature_dim)
        # target = (time_dim, )
        
        # arguments
        self.corpus = corpus
        self.input = np.hstack(input).T
        self.target = np.hstack(target)
        self.sampler = None
                        
    def map_target(self, dct):
        # apply dictionairy mapping
        if type(self.target) == np.ndarray:
            self.target = np.vectorize(dct.get)(self.target)
        elif type(self.target) == torch.Tensor:
            self.target.apply_(dct.get)
        else:      
            print("method expects target of type np.ndarray")
            
    def set_encoding(self, lab2idx=None):
        if lab2idx is None:
            self.labels = sorted(np.unique(self.target))
            self.lab2idx = {lab: i for i, lab in enumerate(self.labels)}
        else:
            self.lab2idx = lab2idx
            
    def encode_target(self, lab2idx=None):
        self.set_encoding(lab2idx)
        self.map_target(self.lab2idx)
        
    def to_tensor(self, mode='classification'):
        if mode == 'classification':
            self.input = torch.tensor(self.input).float()
            self.target = torch.tensor(self.target).long()
        if mode == 'regression':
            self.input = torch.tensor(self.input).float()
            self.target = torch.tensor(self.target).float()  
        
    def to_device(self, device):
        self.input = self.input.to(device)
        self.target = self.target.to(device)
    
    def set_sampler(self, lengths, sampler_args):
        # splicing arguments
        self.sampler_args = sampler_args
        # splicing windows
        splice_idcs = []   
        for length in lengths:
            splice_idcs.extend([self.get_window(frame_idx, length) for frame_idx in range(length)]) 
        # sampler
        self.sampler = {k: v for k, v in enumerate(splice_idcs)}
        
    def get_window(self, frame_idx, nframes):
        # unpack splicing arguments
        n = self.sampler_args['N']
        stride = self.sampler_args['stride']
        # window (clipped to utterance boundaries)
        window = np.arange(-n, n + 1, 1).astype(int) * stride  
        window = np.clip(window, -frame_idx, (nframes - 1)  - frame_idx)
        return window

    def get_input_shape(self):
        return self.__getitem__(0)[0].shape
    
    def get_output_shape(self):
        return self.__getitem__(0)[1].shape

    def split(self, frac=None, seed=None):
        if frac is None:
            return None, self
        else:
            n_split = int(len(self) * frac)
            if seed is not None: torch.manual_seed(seed)
            return torch.utils.data.random_split(self, [n_split, len(self) - n_split])
    
    def __len__(self):
        return len(self.input)
    
    def __getitem__(self, idx):
        # input with sampler
        if self.sampler is None:
            input = self.input[idx]
        else:
            splice_idcs = self.sampler[idx]
            input = self.input[idx + splice_idcs]
            if self.sampler_args['mode'] == 'flatten':
                input = torch.flatten(input, end_dim=1)
            if self.sampler_args['mode'] == 'keep':
                input = input.permute(1, 0) # torch.flatten(input, end_dim=-2)
        # target
        target = self.target[idx]
        return input, target 


    