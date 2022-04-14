#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import numpy as np
import torch
import torch.nn

## Neural network architecure

class FFDNN(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hidden_layer_dims, 
                 nonlinearity=torch.nn.Sigmoid(), dropout=torch.nn.Dropout(0)):
        """
        Fully-connected feedforward deep neural network
        
        Args:
            in_dim:
                The input dimension of the model.
            out_dim:
                The output dimension of the model.
            hidden_layer_dims:
                Hidden layer dimensions of the model.
            nonlinearity:
                Non-linear activations (torch.nn.Module). 
            dropout:
                Dropout layers (torch.nn.Module). 
        """
        
        super().__init__()

        # attributes
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_layer_sizes = hidden_layer_dims
                
        # parameters
        layer_sizes = (in_dim, *hidden_layer_dims, out_dim)
        layer_sizes_pairwise = [(layer_sizes[i], layer_sizes[i+1]) for 
                                 i in range(len(layer_sizes)-1)]

        if type(nonlinearity) is list: self.nonlinearity_layers = nonlinearity
        else: self.nonlinearity_layers = [ nonlinearity for _ in layer_sizes_pairwise ]

        if type(dropout) is list: self.dropout_layers = dropout
        else: self.dropout_layers = [ dropout for _ in layer_sizes_pairwise ]

        # define architecture
        modulelist = torch.nn.ModuleList([])  
        for i, (layer_in_size, layer_out_size) in enumerate(layer_sizes_pairwise):
            # linear layer
            modulelist.append(torch.nn.Linear(layer_in_size, layer_out_size))
            # non-linearity
            if i < len(self.hidden_layer_sizes) and i < len(self.nonlinearity_layers):
                modulelist.append(self.nonlinearity_layers[i])
            # dropout (not between last layer pair)
            if i < len(self.hidden_layer_sizes) and i < len(self.dropout_layers):
                modulelist.append(self.dropout_layers[i])

        # define network as torch.nn.Sequential
        self.net = torch.nn.Sequential(*modulelist)
        
        # initialize weights
        self.apply(init_weights)

    def forward(self, x):
        x = self.net(x)
        return x
    
    def predict(self, inputs):
        outputs = self.net(inputs)
        predictions = torch.argmax(outputs, dim=-1)
        return outputs, predictions



## Modify neural network layers

def init_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        torch.nn.init.xavier_uniform_(m.weight.data)

def set_dropout(m, p=0.1):
    for name, child in m.named_children():
        if isinstance(child, torch.torch.nn.Dropout):
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
         
def train_epoch(model, train_dl, criterion, optimizer, clip_args=None):

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
        if clip_args:
            torch.torch.nn.utils.clip_grad_norm_(model.parameters(), **clip_args)
        # model update
        optimizer.step()
        epoch_loss += loss.item()
        
    return epoch_loss / len(train_dl)
   
def train(model, train_dl, criterion, optimizer, 
          clip_args=None, scheduler=None,
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
        train_loss = train_epoch(model, train_dl, criterion, optimizer, clip_args)
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
                new_lr = scheduler.get_last_lr()
                print("\t -- new lr: %.6f" % (new_lr))        
         
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
    
    # compute error_rate 
    trace = np.trace(confusionmatrix)
    error_rate = 1 - trace.sum() / confusionmatrix.sum()
        
    # compute ER per class (disregarding non label or not)
    n_examples_per_class = confusionmatrix.sum(axis=1)
    error_rate_per_class = [None] * n_classes
    for i in range(n_classes):
        if n_examples_per_class[i] != 0:
            error_rate_per_class[i] = 1 - confusionmatrix[i,i] / n_examples_per_class[i]
        
    return error_rate, error_rate_per_class

def map_labels_cm(confusionmatrix, lab2lab_dict):
    
    n_classes = confusionmatrix.shape[0]
    n_classes_new = len(np.unique((np.fromiter(lab2lab_dict.values(), dtype=int))))
    confusionmatrix_new = np.zeros((n_classes_new, n_classes_new))
    
    # remap classes
    for row in range(n_classes):
        for col in range(n_classes):
            confusionmatrix_new[lab2lab_dict[row], lab2lab_dict[col]] += confusionmatrix[row,col]
            
    return confusionmatrix_new

# Get functions

def get_nonlinearity(nl):
    if nl == 'sig': return torch.nn.Sigmoid()
    if nl == 'relu': return torch.nn.ReLU()
    if nl == 'gelu': return torch.nn.GELU()
    if nl == 'tanh': return torch.nn.Tanh()
    
def get_dropout(dropout_p):
    return torch.nn.Dropout(float(dropout_p))

def get_model(model_super_args):
    
    # fill in arguments
    model_args = model_super_args['model_args'].copy()
    for k, v in model_args.items():
        if k == 'nonlinearity':
            if type(v) is list:
                model_args[k] = [get_nonlinearity(val) for val in v]
            else:
                model_args[k] = get_nonlinearity(v)
        if k == 'dropout': 
            if type(v) is list:
                model_args[k] = [get_dropout(val) for val in v]
            else:
                model_args[k] = get_dropout(v)
    
    # model
    model = None
    model_type = model_super_args['model']
    if model_type == 'ffdnn':
        model = FFDNN(**model_args)
    if model_type == 'tdnn-lstm':
        model = TdnnLstm_icefall(**model_args) 
     
    # if model_type == 'tdnn':
    #     model = CNN(**model_args)    
    # if model_type == 'cnn':
    #     model = CNN(**model_args)   
    
    return model
    
def get_criterion(training_args):
    criterion = None
    if training_args['criterion'] == 'crossentropy':
        criterion = torch.nn.CrossEntropyLoss(**training_args['criterion_args'])
    return criterion

def get_optimizer(training_args, model):
    optimizer = None
    if training_args['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), **training_args['optimizer_args'])
    if training_args['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), **training_args['optimizer_args'])
    if training_args['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), **training_args['optimizer_args'])
    return optimizer

def get_scheduler(training_args, optimizer):
    scheduler = None
    if training_args['scheduler'] == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **training_args['scheduler_args'])    
    return scheduler


class TdnnLstm_icefall(torch.nn.Module):
    def __init__(self, in_dim, out_dim, subsampling_factor) -> None:
        """
        Args:
          num_features:
            The input dimension of the model.
          num_classes:
            The output dimension of the model.
          subsampling_factor:
            It reduces the number of output frames by this factor.
        """
        super().__init__()
        self.num_features = in_dim
        self.num_classes = out_dim
        self.subsampling_factor = subsampling_factor
        self.tdnn = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=self.num_features,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm1d(num_features=512, affine=False),
            torch.nn.Conv1d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm1d(num_features=512, affine=False),
            torch.nn.Conv1d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm1d(num_features=512, affine=False),
            torch.nn.Conv1d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=self.subsampling_factor,  # stride: subsampling_factor!
            ),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm1d(num_features=512, affine=False),
        )
        self.lstms = torch.nn.ModuleList(
            [
                torch.nn.LSTM(input_size=512, hidden_size=512, num_layers=1)
                for _ in range(4)
            ]
        )
        self.lstm_bnorms = torch.nn.ModuleList(
            [torch.nn.BatchNorm1d(num_features=512, affine=False) for _ in range(5)]
        )
        self.dropout = torch.nn.Dropout(0.2)
        self.linear = torch.nn.Linear(in_features=512, out_features=self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x:
            Its shape is [N, C, T]
        Returns:
          The output tensor has shape [N, T, C]
        """
        x = self.tdnn(x)
        x = x.permute(2, 0, 1)  # (N, C, T) -> (T, N, C) -> how LSTM expects it
        for lstm, bnorm in zip(self.lstms, self.lstm_bnorms):
            x_new, _ = lstm(x)
            x_new = bnorm(x_new.permute(1, 2, 0)).permute(
                2, 0, 1
            )  # (T, N, C) -> (N, C, T) -> (T, N, C)
            x_new = self.dropout(x_new)
            x = x_new + x  # skip connections
        x = x.transpose(
            1, 0
        )  # (T, N, C) -> (N, T, C) -> linear expects "features" in the last dim
        x = self.linear(x)
        x = torch.nn.functional.log_softmax(x, dim=-1)
        return x

# class Tdnn(torch.nn.Module):
#     def __init__(self, in_dim, out_dim, hidden_layer_dims, subsampling_factors,
#                  nonlinearity=torch.nn.Sigmoid(), dropout=torch.nn.Dropout(0), 
#                  batchnorm=torch.nn.BatchNorm1d(512)) -> None:
#         """
#         Args:
#           num_features:
#             The input dimension of the model.
#           num_classes:
#             The output dimension of the model.
#           subsampling_factor:
#             It reduces the number of output frames by this factor.
#         """
       
#         super().__init__()

#         # attributes
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.hidden_layer_sizes = hidden_layer_dims
#         self.subsampling_factors = subsampling_factors
        
#          # parameters
#         layer_sizes = (in_dim, *hidden_layer_dims, out_dim)
#         layer_sizes_pairwise = [(layer_sizes[i], layer_sizes[i+1]) for 
#                                  i in range(len(layer_sizes)-1)]
        
#         if type(nonlinearity) is list: self.nonlinearity_layers = nonlinearity
#         else: self.nonlinearity_layers = [ nonlinearity for _ in layer_sizes_pairwise ]

#         if type(dropout) is list: self.dropout_layers = dropout
#         else: self.dropout_layers = [ dropout for _ in layer_sizes_pairwise ]

#         # define architecture
#         modulelist = torch.nn.ModuleList([])  
#         for i, (layer_in_size, layer_out_size) in enumerate(layer_sizes_pairwise):
#             # Conv1D layer
#             modulelist.append(torch.nn.Conv1d(
#                 in_channels=layer_in_size,
#                 out_channels=layer_out_size,
#                 kernel_size=3,
#                 stride=1,
#                 padding=1,
#             ))
#             # non-linearity
#             if i < len(self.hidden_layer_sizes) and i < len(self.nonlinearity_layers):
#                 modulelist.append(self.nonlinearity_layers[i])
#             # dropout (not advisable)
#             if i < len(self.hidden_layer_sizes) and i < len(self.dropout_layers):
#                 modulelist.append(self.dropout_layers[i])
#             # dropout (not between last layer pair)
#             if i < len(self.hidden_layer_sizes) and i < len(self.dropout_layers):
#                 modulelist.append(self.dropout_layers[i])
        



  
# class TDNN(torch.nn.Module):
#     '''
#     Time-delay deep neural network
#     = fully-connected neural network with 1D convolutional layer
#     ''' 
#     def __init__(self, in_dim, out_dim, hidden_layer_sizes,
#                  kernel_size=(5, ), padding=(2, ), n_filters=100,
#                  cnn_position=0):
#         super(TDNN, self).__init__()

#         # attributes
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.hidden_layer_sizes = hidden_layer_sizes

#         # parameters
#         layer_sizes = (in_dim, *hidden_layer_sizes, out_dim)
#         layer_sizes_pairwise = [(layer_sizes[i], layer_sizes[i+1]) for 
#                                  i in range(len(layer_sizes)-1)]

#         # define architecture
#         modulelist = torch.nn.ModuleList([])
#         for i, (layer_in_size, layer_out_size) in enumerate(layer_sizes_pairwise):

#             if i == cnn_position:
#                 modulelist.append(torch.nn.Conv1d(layer_in_size, n_filters, 
#                                             kernel_size, padding=padding))
#                 modulelist.append(View(shape=(-1, )))
                    
#             else:
#                 modulelist.append(torch.nn.Linear(layer_in_size, layer_out_size))

#             if i < len(self.hidden_layer_sizes):
#                 modulelist.append(torch.nn.Sigmoid())

#         # define network as torch.nn.Sequential
#         self.net = torch.nn.Sequential(*modulelist)

#     def forward(self, x):
#         x = self.net(x)
#         return x
    
#     def predict(self, inputs):
#         outputs = self.net(inputs)
#         predictions = torch.argmax(outputs, dim=-1)
#         return outputs, predictions

# class CNN(torch.nn.Module):
#     '''
#     Convolutional neural network
#     = fully-connected neural network with 2D convolutional input layer
#     '''
#     def __init__(self, in_dim, out_dim, hidden_layer_sizes,
#                  kernel_size=(5, 5), padding=(2, 2), n_filters=25,
#                  cnn_position=0):
#         super(CNN, self).__init__()

#         # attributes
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.hidden_layer_sizes = hidden_layer_sizes

#         # parameters
#         layer_sizes = (in_dim, *hidden_layer_sizes, out_dim)
#         layer_sizes_pairwise = [(layer_sizes[i], layer_sizes[i+1]) for 
#                                  i in range(len(layer_sizes)-1)]

#         # define architecture
#         modulelist = torch.nn.ModuleList([])
#         for i, (layer_in_size, layer_out_size) in enumerate(layer_sizes_pairwise):

#             if i == cnn_position:
#                 modulelist.append(torch.nn.Conv2d(layer_in_size, n_filters, 
#                                             kernel_size, padding=padding))
#                 modulelist.append(View(shape=(-1, )))
                    
#             else:
#                 modulelist.append(torch.nn.Linear(layer_in_size, layer_out_size))

#             if i < len(self.hidden_layer_sizes):
#                 modulelist.append(torch.nn.Sigmoid())

#         # define network as torch.nn.Sequential
#         self.net = torch.nn.Sequential(*modulelist)

#     def forward(self, x):
#         x = self.net(x)
#         return x

# # view: (B, ...) -> (B, shape)
# class View(torch.nn.Module):
    
#     def __init__(self, shape):
#         super().__init__()
#         self.shape = shape

#     def forward(self, input):
#         batch_size = input.size(0)
#         shape = (batch_size, *self.shape)
#         out = input.view(shape)
#         return out