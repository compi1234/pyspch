#  Utilities
#
#
import os,sys,io 

from IPython.display import display, Audio, HTML, clear_output

import math
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 

from pyspch.constants import EPS_FLOAT, LOG10, SIGEPS_FLOAT


##################################################################################################
# PART I:  MATH UTILITIES
##################################################################################################
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
        

