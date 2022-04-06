#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import Counter
import torch
import torch.nn as nn
import numpy as np

## Neural network architecure

class FFDNN(nn.Module):
    '''
    fully-connected feedforward deep neural network
    '''
    def __init__(self, in_dim, out_dim, hidden_layer_dims, 
                 nonlinearity=nn.Sigmoid(), dropout=nn.Dropout(0)):
        super(FFDNN, self).__init__()

        # attributes
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_layer_sizes = hidden_layer_dims
        self.nonlinearity = nonlinearity
        self.dropout = dropout
        
        # parameters
        layer_sizes = (in_dim, *hidden_layer_dims, out_dim)
        layer_sizes_pairwise = [(layer_sizes[i], layer_sizes[i+1]) for 
                                 i in range(len(layer_sizes)-1)]

        # define architecture
        modulelist = nn.ModuleList([])  
        for i, (layer_in_size, layer_out_size) in enumerate(layer_sizes_pairwise):
            modulelist.append(nn.Linear(layer_in_size, layer_out_size))
            if i < len(self.hidden_layer_sizes):
                modulelist.append(self.nonlinearity)
                modulelist.append(self.dropout)

        # define network as nn.Sequential
        self.net = nn.Sequential(*modulelist)
        
        # initialize weights
        self.apply(init_weights)

    def forward(self, x):
        x = self.net(x)
        return x
    
class TDNN(nn.Module):
    '''
    Time-delay deep neural network
    = fully-connected neural network with 1D convolutional layer
    ''' 
    def __init__(self, in_dim, out_dim, hidden_layer_sizes,
                 kernel_size=(5, ), padding=(2, ), n_filters=100,
                 cnn_position=0):
        super(TDNN, self).__init__()

        # attributes
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_layer_sizes = hidden_layer_sizes

        # parameters
        layer_sizes = (in_dim, *hidden_layer_sizes, out_dim)
        layer_sizes_pairwise = [(layer_sizes[i], layer_sizes[i+1]) for 
                                 i in range(len(layer_sizes)-1)]

        # define architecture
        modulelist = nn.ModuleList([])
        for i, (layer_in_size, layer_out_size) in enumerate(layer_sizes_pairwise):

            if i == cnn_position:
                modulelist.append(nn.Conv1d(layer_in_size, n_filters, 
                                            kernel_size, padding=padding))
                modulelist.append(View(shape=(-1, )))
                    
            else:
                modulelist.append(nn.Linear(layer_in_size, layer_out_size))

            if i < len(self.hidden_layer_sizes):
                modulelist.append(nn.Sigmoid())

        # define network as nn.Sequential
        self.net = nn.Sequential(*modulelist)

    def forward(self, x):
        x = self.net(x)
        return x

class CNN(nn.Module):
    '''
    Convolutional neural network
    = fully-connected neural network with 2D convolutional input layer
    '''
    def __init__(self, in_dim, out_dim, hidden_layer_sizes,
                 kernel_size=(5, 5), padding=(2, 2), n_filters=25,
                 cnn_position=0):
        super(CNN, self).__init__()

        # attributes
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_layer_sizes = hidden_layer_sizes

        # parameters
        layer_sizes = (in_dim, *hidden_layer_sizes, out_dim)
        layer_sizes_pairwise = [(layer_sizes[i], layer_sizes[i+1]) for 
                                 i in range(len(layer_sizes)-1)]

        # define architecture
        modulelist = nn.ModuleList([])
        for i, (layer_in_size, layer_out_size) in enumerate(layer_sizes_pairwise):

            if i == cnn_position:
                modulelist.append(nn.Conv2d(layer_in_size, n_filters, 
                                            kernel_size, padding=padding))
                modulelist.append(View(shape=(-1, )))
                    
            else:
                modulelist.append(nn.Linear(layer_in_size, layer_out_size))

            if i < len(self.hidden_layer_sizes):
                modulelist.append(nn.Sigmoid())

        # define network as nn.Sequential
        self.net = nn.Sequential(*modulelist)

    def forward(self, x):
        x = self.net(x)
        return x

# view: (B, ...) -> (B, shape)
class View(nn.Module):
    
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, input):
        batch_size = input.size(0)
        shape = (batch_size, *self.shape)
        out = input.view(shape)
        return out

## Modify neural network layers

def init_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

def set_dropout(m, p=0.1):
    for name, child in m.named_children():
        if isinstance(child, torch.nn.Dropout):
            child.p = p
        set_dropout(child, drop_rate=p)

## Neural network training 

class EarlyStop(object):
    
    def __init__(self, patience):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
    
    def update(self, loss):
        if loss < self.best_loss:
            self.counter = 0
            self.best_loss = loss
        else:
            self.counter += 1  
    
    def stop(self):
        return self.counter > self.patience
         
def train_epoch(model, train_dl, criterion, optimizer, clip=None):

    model.train()
    epoch_loss = 0

    # iterate over batches 
    for batch in iter(train_dl):
        # forward pass
        optimizer.zero_grad()
        inputs, targets = batch
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # backward pass
        loss.backward()
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        # model update
        optimizer.step()
        epoch_loss += loss.item()
        
    return epoch_loss / len(train_dl)
   
def train(model, train_dl, criterion, optimizer, 
          clip=None, scheduler=None,
          current_epoch=0, n_epochs=500, 
          valid_dl=None, patience=5, every=10,):

    '''mini-batch gradient descent with early stopping'''

    # current_epoch
    start_epoch = end_epoch = current_epoch

    # training and validation loss
    train_losses = []
    valid_losses = []
    
    # train 
    earlystop = EarlyStop(patience)
    for epoch in np.arange(start_epoch, start_epoch + n_epochs):
        
        # train epoch
        train_loss = train_epoch(model, train_dl, criterion, optimizer, clip)
        train_losses.append(train_loss)
        end_epoch = epoch
        if epoch % every == 0:   
            print("Epoch %d -- av. train loss per mini-batch %.2f" % (epoch, train_loss))
          
        # early stoppping
        if valid_dl is not None:
            valid_loss = evaluate(model, valid_dl, criterion)
            valid_losses.append(valid_loss)
            if epoch % every == 0:   
                print("\t -- av. validation loss per mini-batch %.2f" % (valid_loss)) 
            earlystop.update(valid_loss)
            if earlystop.stop():
                print("\t -- stop early")
                break

        # update scheduler
        if scheduler is not None:
            if type(scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                metrics = train_loss if valid_dl is None else valid_loss
                scheduler.step(metrics)
            else:
                scheduler.step()           
         
    return train_losses, valid_losses, end_epoch

## Evaluation with DataLoader (iterator)

def evaluate(model, iterator, criterion):

    model.eval()
    running_loss = 0

    # compute average loss
    with torch.no_grad():
        for batch in iter(iterator):
            # forward pass
            inputs, targets = batch
            outputs = model(inputs)
            # loss
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            
    average_loss = running_loss / len(iterator) 

    return average_loss
        
def evaluate_cm(model, iterator):

    model.eval()
    cm = np.zeros((model.out_dim, model.out_dim))
    
    # compute confusion matrix
    with torch.no_grad():
        for batch in iter(iterator):
            # forward pass
            inputs, targets = batch
            outputs = model(inputs)
            # prediction 
            predictions = torch.argmax(outputs, dim=-1) 
            for target, prediction in zip(targets, predictions):
                cm[target][prediction] += 1
    
    return cm.astype(int)

## Evaluation with arrays

def evaluate_array(model, inputs, targets, criterion, device, 
                   mode="classification"):
    
    model.eval()
    
    # format array to torch tensor 
    if mode == 'classification':
        inputs = torch.tensor(inputs, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.long)
    if mode == 'regression':
        inputs = torch.tensor(inputs, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)
        
    # send to device (ex. GPU)  
    inputs.to(device)
    targets.to(device)
    model.to(device)

    # evaluate loss based on criterion
    outputs = model(inputs)
    loss = criterion(outputs, targets).item()

    return loss

def evaluate_cm_array(model, inputs, targets, device,
                      n_per_pass=2**10, unsqueeze=False, unsqueeze2d=False):
    
    model.eval()
    
    # format array to torch tensor + send to device (ex. GPU)
    # confusion matrix assumes 'classification' -> y.dtype = torch.long
    inputs = torch.tensor(inputs, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.long)

    # send to device (ex. GPU)  
    inputs.to(device)
    targets.to(device)
    model.to(device)
    
    # compute confusion matrix
    cm = np.zeros((model.out_dim, model.out_dim))
        
    # split tensor in managable chunks
    n_samples = len(inputs)
    loop_n_times = math.ceil(n_samples/n_per_pass)
    
    # iterate over chunks
    for i in range(loop_n_times):
        inputs_chunks = inputs[i*n_per_pass:(i+1)*n_per_pass]
        targets_chunks = targets[i*n_per_pass:(i+1)*n_per_pass]
        for input, target in zip(inputs_chunks, targets_chunks):
            # outputs (posterior class probabilities)
            if unsqueeze: output = model(torch.unsqueeze(input, 0))
            elif unsqueeze2d: output = model(torch.reshape(input, (1, *input.shape)))
            else: output = model(input)  
            # labels (assumes One-Hot Encoding)
            label = torch.argmax(output) 
            cm[target][label] += 1
        
    return cm.astype(int)

## Confusion matrix auxiliaries

def cm2per(confusionmatrix):

    n_classes = confusionmatrix.shape[0]    
    
    # compute ER 
    trace = np.trace(confusionmatrix)
    ER = 1- trace.sum() / confusionmatrix.sum()
        
    # compute ER per class (disregarding non label or not)
    no_examples_pc = confusionmatrix.sum(axis=1)
    ER_pc = [None] * n_classes
    for i in range(n_classes):
        if no_examples_pc[i] != 0:
            ER_pc[i] = 1-confusionmatrix[i,i] / (no_examples_pc[i])
        
    return ER, ER_pc

def map_labels_cm(confusionmatrix, lab2lab_dict):
    
    n_classes = confusionmatrix.shape[0]
    n_classes_new = len(np.unique((np.fromiter(lab2lab_dict.values(), dtype=int))))
    confusionmatrix_new = np.zeros((n_classes_new, n_classes_new))
    
    # remap classes
    for row in range(n_classes):
        for col in range(n_classes):
            confusionmatrix_new[lab2lab_dict[row], lab2lab_dict[col]] += confusionmatrix[row,col]
            
    return confusionmatrix_new

