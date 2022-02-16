""" Utilities for feature manipulation """

import math
import numpy as np
import librosa
#from .constants import EPS_FLOAT


def mean_norm(ftrs,type="mean"):
    """ normalizes features for mean and variance depending on Norm argument 
    arguments:
        ftrs:   nd.array of size [n_ftrs,n_frames] or [n_frames]
        type:   type of normalization
                    "mean": mean normalization
                    "var":  variance normalization
                    "meanvar": mean and variance normalization
    returns:
        ftrs:   nd.array of size [n_ftrs,n_frames]
    """
    ftrs = np.atleast_2d(ftrs)
    if type is None:
        ftrs_n = ftrs
    elif type == "mean":
        ftrs = (ftrs - ftrs.mean(axis=1,keepdims=True) )
    elif type == "var":
        ftrs = ftrs / ftrs.std(axis=1,keepdims=True)  
    elif type == "meanvar":
        ftrs = (ftrs - ftrs.mean(axis=1,keepdims=True) )/ ftrs.std(axis=1,keepdims=True)     
    else:
        print("WARNING(mean_norm): Normalization(%s) not recognized",Norm)
        
    return(ftrs)


def deltas(ftrs,type="delta",Augment=False):
    '''
    deltas() provides a set of delta-processing functions
    
    arguments:
        ftrs:   nd.array of size [n_ftrs,n_frames]
        type:   type of processing
                    "delta":  first order delta       filter coeff's: [ -2 -1 0 1 2]
                    "delta2": second order delta      filter coeff's: [1 -2 1]
                    "delta_delta2": first and second order delta
        augment: False (default)
                    output will return the delta features
                 True
                    output will augment the input features with the delta features
    returns:
        output:   nd.array of size [n_ftrs_out,n_frames]
                    output feature dimension depending on processing
    '''
    
    if type == "delta":
        ftrs_out = librosa.feature.delta(ftrs, width=5, order=1, axis=- 1, mode='nearest')
    elif type == "delta2":
        ftrs_out = librosa.feature.delta(ftrs, width=3, order=2, axis=- 1, mode='nearest')
    elif type == "delta_delta2":
        ftrs_out = np.vstack((
            librosa.feature.delta(ftrs, width=5, order=1, axis=- 1, mode='nearest'),
            librosa.feature.delta(ftrs, width=3, order=2, axis=- 1, mode='nearest')
            ))
    else: ftrs_out=ftrs
    if Augment:
        return( np.vstack( (ftrs,ftrs_out) ) )
    else:
        return(ftrs_out)




def pad_frames(X,N = 1):
    '''
    padding copies the edge frames N times 
    e.g.   padding sequence [a,b,c,d] with 2 frames will result in 
                      [a,a,a,b,c,d,d,d]    
    '''
    X_padded = np.pad(X,[(0,0),(N,N)],mode='edge')
    return X_padded



def splice_frames(X,N=1,stride=1):
    '''
    symmetrically splice 2*N+1 frames , i.e. add N frames on left and right side
    to the current feature vector, each time shifted by 'stride' frames
    '''
    NN = N*stride
    X_p = pad_frames(X,N=NN)
    X_s = X
    n_frames = X.shape[1]
    for i in range(1,N+1):
        ii = i*stride
        X_s = np.concatenate((
            X_p[:,NN-ii:NN-ii+n_frames],
            X_s,
            X_p[:,NN+ii:NN+ii+n_frames]
            ),axis=0)
    return X_s    
    
    
    
    
    
    
    

def feature_extraction(wavdata=None, spg=None,n_mels=80,sample_rate=8000,ncep=13,Deltas=True,Norm=None):
    #1. Spectral Estimation (Fourier Spectrogram)
    if spg is None:
        spg = Sps.spectrogram(wavdata,sample_rate=sample_rate,n_mels=None)
    
    #2. Mel scale transform
    if n_mels is not None:
        spg1 = Sps.spg2mel(spg,n_mels=n_mels,sample_rate=sample_rate)
    else: spg1 = spg
        
    #3. Cepstral transform
    if ncep is None: ftr1 = spg1
    else: ftr1 = cepstrum(spg1,ncep=ncep)
        
    #4. Add delta's
    if Deltas:
        deltas = librosa.feature.delta(ftr1,order=1)
        ftrs = np.vstack((ftr1,deltas))
    else: ftrs = ftr1
        
    #5. Mean and variance normalization

    return(ftrs_n)    