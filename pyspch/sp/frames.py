""" Utilities for frame handling, syncrhonization, ... """
import math
import numpy as np
import librosa
from ..core.constants import EPS_FLOAT


# do preemhasis and padding
def preemp_pad(y,pad=None,preemp=None):
    '''
    constructs an array that is padded and preemphasized
    to make the output suitable for librosa stft processing with center = False (with a predictable number of frames)
        - for usage with librosa stft set pad to (n_fft-n_shift)/2
        - for time domain based feature extraction set pad to (n_length-n_shift)/2
    '''
    if pad is None: y_padded =y
    else: y_padded = np.concatenate((y[0:pad][::-1],y,y[:-pad-1:-1]))
    if preemp is None: 
        z = y_padded
    else:
        z=y_padded.copy()
        z[0]=(1.-preemp)*z[0]
        z[1:]= z[1:] - preemp*z[0:-1]
    return(z)

def make_frames(y,pad=None,preemp=None,n_shift=80,n_length=240,window='hamm'):
    ''' 
    converts a 1D signal array to a 2D array of frames
    with appropriate shift, length, windowing, preemphasis and padding 
    
    can be used as the 'framing operation' for any frame based processing
    '''
    if pad is True:  pad = (n_length-n_shift)//2
    y_pre = preemp_pad(y,pad=pad,preemp=preemp)
    nfr = (len(y_pre)-n_length)//n_shift + 1
    frames = np.zeros((n_length,nfr),dtype=y.dtype)
    if window==None:
        y_window = np.ones(n_length)
    else:
        y_window = librosa.filters.get_window(window,n_length)
    for i in range(nfr):
        yy = y_pre[i*n_shift:(i*n_shift+n_length)]
        frames[:,i] = [yy[j] * y_window[j] for j in range(n_length)]
    return(frames)


##################################################################################################
# General Purpose utilities, handy in time-frequency processing
##################################################################################################
# time to index conversions; 
# - for standard sampling use default frames=False
#      ti = i * dt
# - for sampling of frames, you may prefer frames=True placing
#     ti = (i+0.5) * dt    
#
# inputs can be scalars, lists or numpy arrays  outputs are always numpy arrays
def t2indx(t,dt=1.,Frames=False):
    """ time-to-index conversion:  ; see indx2t() for details"""
    offs = 0.0 if Frames == False else 0.5
    return np.round((np.array(t).astype(float)/float(dt)-offs)).astype(int)

def indx2t(i,dt=1.,Frames=False):
    """ index-to-time conversion: 
        time[i] = (i+offs) * dt  ; offs=0.5 when 'center'(default)
        
    dt : sampling period
    Frames : default(=False) """
    offs = 0.0 if Frames == False else 0.5
    return (np.array(i).astype(float) + offs )*dt

def time_range(n,dt=1.,Frames=False):
    """ indx2t() for n samples 0 ... n-1 """
    offs = 0.0 if Frames == False else 0.5
    return (np.arange(n,dtype='float32')+offs)*dt 

