""" Time domain feature extraction  """
import math
import numpy as np
import librosa
from .frames import preemp_pad

# time domain feature extraction from librosa
# 
def time_dom3(y,shift=0.01,length=0.03,sr=8000,pad=True,preemp=0.0):
    '''
    computes 3 time domeain features: RMS energy, pitch and zero crossing rate
    the framing and padding is done as in the Sps spectrogram routines
    
    arguments
        y      waveform data
        shift, length   frame shift and length in sec's  (defaults  0.01 and 0.03)
        sr     sample rate (default 8000)
        pad    boolean for padding (default is True)
        preemp default = 0.0
        
    returns rms  (in amplitude)
            pitch (in Hz)
            zcr   (rate is 'per second')
    '''
    n_shift = int(shift*sr)
    n_length = int(length*sr)
    if pad is True:  pad = (n_length-n_shift)//2
    y1 = preemp_pad(y,pad=pad,preemp=preemp)
    zcr = librosa.feature.zero_crossing_rate(y=y1,frame_length=n_length,hop_length=n_shift,center=False)
    pitch = librosa.pyin(y=y1,frame_length=n_length,hop_length=n_shift,center=False,
                               sr=sr, fmin = 50., fmax=450.) 
    rms = librosa.feature.rms(y=y1,frame_length=n_length,hop_length=n_shift,center=False)      
    return(rms,pitch[0],zcr/shift)