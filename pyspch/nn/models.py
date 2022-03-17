#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
import torch
import torch.nn as nn
import numpy as np

## Neural network architecure

class FFDNN(nn.Module):
    """
    fully-connected feedforward deep neural network
    """
    def __init__(self, in_dim, out_dim, hidden_layer_dims):
        super(FFDNN, self).__init__()

        # attributes
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_layer_sizes = hidden_layer_dims

        # parameters
        layer_sizes = (in_dim, *hidden_layer_dims, out_dim)
        layer_sizes_pairwise = [(layer_sizes[i], layer_sizes[i+1]) for 
                                 i in range(len(layer_sizes)-1)]

        # define architecture
        modulelist = nn.ModuleList([])  
        for i, (layer_in_size, layer_out_size) in enumerate(layer_sizes_pairwise):
            modulelist.append(nn.Linear(layer_in_size, layer_out_size))
            if i < len(self.hidden_layer_sizes):
                modulelist.append(nn.Sigmoid())

        # define network as nn.Sequential
        self.net = nn.Sequential(*modulelist)
        
        # initialize weights
        self.apply(weight_init)

    def forward(self, x):
        x = self.net(x)
        return x
    
    def train_epoch(self, iterator, criterion, optimizer, clip=None):

        self.train()
        epoch_loss = 0
    
        # iterate over batches 
        for batch in iter(iterator):
            # forward pass
            optimizer.zero_grad()
            inputs, targets = batch
            outputs = self(inputs)
            loss = criterion(outputs, targets)
            # backward pass
            loss.backward()
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
            # model update
            optimizer.step()
            epoch_loss += loss.item()
            
        return epoch_loss / len(iterator)
     
    def evaluate(self, iterator, criterion):

        self.eval()
        running_loss = 0
    
        # compute average loss
        with torch.no_grad():
            for batch in iter(iterator):
                # forward pass
                inputs, targets = batch
                outputs = self(inputs)
                # loss
                loss = criterion(outputs, targets)
                running_loss += loss.item()
                
        average_loss = running_loss / len(iterator) 
    
        return average_loss
            
    def evaluate_cm(self, iterator):

        self.eval()
        cm = np.zeros((self.out_dim, self.out_dim))
        
        # compute confusion matrix
        with torch.no_grad():
            for batch in iter(iterator):
                # forward pass
                inputs, targets = batch
                outputs = self(inputs)
                # prediction 
                predictions = torch.argmax(outputs, dim=-1) 
                for target, prediction in zip(targets, predictions):
                    cm[target][prediction] += 1
        
        return cm.astype(int)
     
class TDNN(nn.Module):
    """
    Time-delay deep neural network
    = fully-connected neural network with 1D convolutional layer
    """ 
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
    """
    Convolutional neural network
    = fully-connected neural network with 2D convolutional input layer
    """
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

## Initialization

def weight_init(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

## Neural network training

# mini-batch gradient descent with early stopping 

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
            self.counter += 0  
    
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
   
def train(model, train_dl, criterion, optimizer, clip=None, 
          current_epoch=0, n_epochs=500, 
          valid_dl=None, patience=5, every=10,):

    # current_epoch
    start_epoch = end_epoch = current_epoch

    # training and validation loss
    train_losses = []
    valid_losses = []
    
    # train 
    earlystop = EarlyStop(patience)
    for epoch in np.arange(start_epoch, start_epoch + n_epochs):
        
        # train epoch
        train_loss = model.train_epoch(train_dl, criterion, optimizer, clip)
        train_losses.append(train_loss)
        end_epoch = epoch
        
        # print
        if epoch % every == 0:   
            print("Epoch %d -- av. train loss per mini-batch %.2f" % (epoch, train_loss))
            
        # early stoppping
        if valid_dl is not None:
            valid_loss = model.evaluate(valid_dl, criterion)
            valid_losses.append(valid_loss)
            if epoch % every == 0:   
                print("\t -- av. validation loss per mini-batch %.2f" % (valid_loss))

            earlystop.update(valid_loss)
            if earlystop.stop():
                break
            
    return train_losses, valid_losses, end_epoch

## Evaluation

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


