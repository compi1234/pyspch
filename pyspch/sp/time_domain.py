""" Time domain feature extraction  """
import math
import numpy as np
import librosa
from .frames import preemp_pad
from ..core.utils import seconds_to_samples

# time domain feature extraction from librosa
# 
def time_dom3(y,shift=0.01,length=0.03,sr=8000,pad=True,preemp=0.0):
    '''
    computes 3 time domain features: RMS energy, pitch and zero crossing rate
    the framing and padding is done as in the Sps spectrogram routines
    
    arguments
    ---------
        y      waveform data
        shift  frame shift in sec's (default = 0.01)
        length frame length in sec's  (default = 0.03)
        sr     sample rate (default 8000)
        pad    boolean for padding (default is True)
        preemp default = 0.0
        
    returns
    -------
        rms   (in amplitude)
        pitch (in Hz)
        zcr   (rate is 'per second')
            
    Note: convert RMS to energy per sample
        E (per sample) = rms**2
        Energy in dB:  10*log10(E)
    '''
    n_shift = seconds_to_samples(shift,sr)
    n_length = seconds_to_samples(length,sr)
    if pad is True:  pad = (n_length-n_shift)//2
    if pad < 0:
        print("WARNING(time_dom3): length < shift, EXPECT WEIRD RESULTS !!")
        pad = 0
    y1 = preemp_pad(y,pad=pad,preemp=preemp)
    zcr = librosa.feature.zero_crossing_rate(y=y1,frame_length=n_length,hop_length=n_shift,center=False)
    pitch = librosa.pyin(y=y1,frame_length=n_length,hop_length=n_shift,center=False,
                               sr=sr, fmin = 50., fmax=450.) 
    rms = librosa.feature.rms(y=y1,frame_length=n_length,hop_length=n_shift,center=False)      
    return(rms,pitch[0],zcr/shift)


def energy(y,sr=8000,shift=0.01,length=0.03,pad=True,preemp=0.0,mode='dB'):
    n_shift = seconds_to_samples(shift,sr)
    n_length = seconds_to_samples(length,sr)
    if pad is True:  pad = (n_length-n_shift)//2
    y1 = preemp_pad(y,pad=pad,preemp=preemp)   
    
    rms = librosa.feature.rms(y=y1,frame_length=n_length,hop_length=n_shift,center=False)   

    mode = mode.lower()
    if mode == 'magnitude':
        return(rms)
    elif mode == 'power':
        return(rms**2)
    elif mode == 'db':
        return(10.*np.log10(rms**2))