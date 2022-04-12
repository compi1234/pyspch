#  Utilities
#
#
import os,sys,io 

import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 

from .constants import EPS_FLOAT, LOG10, SIGEPS_FLOAT

##################################################
# PART 0:  GLOBAL UTILITIES
##################################################
####
# A single call to check if IN_COLAB has been set
####
def check_colab():
    return('google.colab' in str(get_ipython()))

##################################################
# PART I:  MATH UTILITIES
##################################################

def next_power_of_2(x):  
    return 1 if x == 0 else 2**(x - 1).bit_length()

def normalize(x, axis=0):
    """Normalizes a multidimensional input array so that the values sums to 1 along the specified axis
    Typically applied to some multinomal distribution

    x       numpy array
            of not normalized data
    axis    int
            dimension along which the normalization should be done

    """

    xs = x.sum(axis,keepdims=True)
    xs[xs<EPS_FLOAT]=1.
    shape = list(x.shape)
    shape[axis] = 1
    xs.shape = shape
    return(x / xs)

def floor(x,FLOOR=EPS_FLOAT):
    """ array floor:  returns  max(x,FLOOR)  """
    return(np.maximum(x,FLOOR))

def logf(x,eps=EPS_FLOAT):
    """ array log with flooring """
    return(np.log(np.maximum(x,eps)))
    
def log10f(x,eps=EPS_FLOAT):
    """ array log10 with flooring """
    return(np.log10(np.maximum(x,eps)))
    
def convertf(x,iscale="lin",oscale="log",eps=EPS_FLOAT):
    """ array conversions between lin, log and log10 with flooring protection """
    if iscale == oscale: 
        return x
    
    if iscale == "lin":
        if oscale == "log":
            return logf(x,eps)
        elif oscale == "log10":
            return log10f(x,eps)
    elif iscale == "log":
        if oscale == "lin":
            return np.exp(x)
        elif oscale == "log10":
            return x/LOG10
    elif iscale == "log10":
        if oscale == "lin":
            return np.power(10.0,x)
        elif oscale == "log":
            return x*LOG10

def dct_diff(dct_a, dct_b):
    dct_a = {k: v for k, v in dct_a.items() if v is not None}
    dct_b = {k: v for k, v in dct_b.items() if v is not None}
    return { k : dct_b[k] for k in set(dct_b.keys()) - set(dct_a.keys()) }
        
##################################################
# PART II: Labels and Segmentations
##################################################
def lbls2seg(lbls,shift=0.01):
    '''
    converts a label sequence (alignment) to a segmentation DataFrame
    This conversion may fail if some frames are not labeled
    
    Arguments:
    lbls       np.array of labels
    shift      frame shift (default=0.01)
    
    Output:
    segmentation dataframe with columns {'t0','t1','seg'}
    '''

    t1 = []
    t0 = [ 0. ]
    seg = [ lbls[0] ]
    for i in range(1,len(lbls)):
        if lbls[i] != lbls[i-1]:
            t0.append( i*shift )
            seg.append(lbls[i])
            t1.append( i*shift )
    t1.append( len(lbls) * shift )
    return( pd.DataFrame({'t0':t0,'t1':t1,'seg':seg}) )   

def seg2lbls(seg, shift=0.01, n_frames=None, end_time=None, pad_lbl=None):
    '''
    seg2lbls() converts a segmentation DataFrame to an alignment 
                in the form of an array of labels
                This conversion will fill empty regions with the padding label
                'pad_lbl' and will make an output at least as large as n_frames (or end_time)
    
    Arguments:
    seg      Segmentation DataFrame with columns {'t0','t1','seg'}
    shift    frame shift (in units used in seg)
    n_frames desired number of frames, overrides last value in seg if larger
    end_time desired end time of segmentation, overrides last value in seg if larger
    pad_lbl  a padding label for missing segments (default=None)
    
    Returns:
    lbls[]   list with labels
    
    '''

    # determine desired end_time/n_frames from arguments or segmentation
    if (n_frames is None) and (end_time is not None):
        n_frames = round(end_time/shift)

    n_fr = round(seg['t1'].iloc[-1]/shift)
    if n_frames is None: 
        n_frames = n_fr
    # else: n_frames = max(n_frames,n_fr)
        
    lbls = [pad_lbl] * n_frames
    for _, seg in seg.iterrows():
        f0 = max(round(seg['t0']/shift), 0)
        f1 = min(round(seg['t1']/shift), n_frames)
        lbls[f0:f1] = [seg['seg']] * int(f1 - f0)
        
    return lbls 


##################################################
# PART III: Confusion Matrices
##################################################

# Pretty Print routine makes for confusion matrices
def plot_confusion_matrix(cm,labels=[],title='Confusion Matrix\n',figsize=(4,4),**kwargs):
    '''
    Plot a Confusion Matrix
    
    Arguments
        cm        confusion matrix, np.array or DataFrame
        labels    Default = []
        title     Default = 'Confusion Matrix'
        figsize   Default = (4,4)
        **kwargs  extra arguments to pass to sns.heatmap()
    '''
    
    heatmap_args = {
        'annot': True, 
        'fmt': 'd', 
        'annot_kws': {'fontsize': 12, 'color':'k'},
        'xticklabels':labels,
        'yticklabels':labels,
        'square': True, 
        'linecolor': 'k', 
        'linewidth': 1.5, 
        'cmap':'Blues', 
        'cbar': False
        }
    
    heatmap_args.update(kwargs)
    f,ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, **heatmap_args)
    ax.tick_params(axis='y',labelrotation=0.0,left=True)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    return f,ax
