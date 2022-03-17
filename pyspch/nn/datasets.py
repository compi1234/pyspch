#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.utils.data import Dataset

### DataSet ###

class SpchDataset(Dataset):
    
    def __init__(self, corpus, input, target):
        
        # arguments
        self.corpus = corpus
        self.input = np.hstack(input).T
        self.target = np.hstack(target)
    
    def set_encoding(self, lab2idx=None):
        if lab2idx is None:
            self.labels = sorted(np.unique(self.target))
            self.lab2idx = {lab: i for i, lab in enumerate(self.labels)}
        else:
            self.lab2idx = lab2idx

    def encode_target(self, lab2idx=None):
        # set enocding
        self.set_encoding(lab2idx)
        # encode target
        if type(self.target) == np.ndarray:
            self.target = np.vectorize(self.lab2idx.get)(self.target)
        elif type(self.target) == torch.Tensor:
            self.target.apply_(self.lab2idx.get)
        else:      
            print("method expects target of type np.ndarray")
        
    def to_tensor(self):
        self.input = torch.tensor(self.input)
        self.target = torch.tensor(self.target).long()
        
    def to_device(self, device):
        self.input = self.input.to(device)
        self.target = self.target.to(device)
       
    def set_sampler(self, lengths, splice_args):
        # splicing arguments
        self.splice_args = splice_args
        # splicing windows
        splice_idcs = []   
        for length in lengths:
            splice_idcs.extend([self.get_window(frame_idx, length) for frame_idx in range(length)]) 
        # sampler
        self.sampler = {k: v for k, v in enumerate(splice_idcs)}
        
    def get_window(self, frame_idx, nframes):
        # unpack splicing arguments
        n = self.splice_args['N']
        stride = self.splice_args['stride']
        # window (clipped to utterance boundaries)
        window = np.arange(-n + 1, n, 1).astype(int) * stride  
        window = np.clip(window, -frame_idx, nframes - frame_idx)
        return window

    def __len__(self):
        return len(self.input)
    
    def __getitem__(self, idx):
        splice_idcs = self.sampler[idx]
        input = torch.flatten(self.input[splice_idcs], end_dim=1)
        label = self.target[idx]
        return input, label 


