""" Utilities for signal generation """

import math
import numpy as np
from scipy import signal
from .spectral import *

def synth_fourier(freqs = [200.], amps = [1.], phis = None, dur = 0.5,sample_rate=8000.):
    '''
    Fourier synthesis for given frequencies, amplitudes and phases
    
    Amplitude and phase can be given globally or per component
    
    Input Parameters:
    -----------------
    freqs     array like
        frequencies to be synthesized
    amps      array like (single float or of same dim as freqs)
        amplitude(s) of synthesized components
    phis      array like (single float or of same dim as freqs)
        phase(s) of synthesized components
    dur :     float (default=0.5)
        duration in seconds
    sample_rate: float (default=8000.)
    
    Returns:
    --------
    y    np.array of floats
        synthesized signal
    t    np.array of floats
        sample times (in seconds, starting at 0.0)
    '''
    
    freqs = np.asarray(freqs)
    amps = np.asarray(amps)
    #
    N = len(freqs)
    if  len(amps) != N: amps = amps[0]*np.ones(N)
    if phis is None: phis = np.zeros(N)
    elif len(phis) != N : phis = phis[0]*np.ones(N)       
    t = np.linspace(0.0, dur, int(dur*sample_rate), endpoint=False)
    y = 0*t
    for k in range(N):
        y = y+ amps[k]*np.sin( 2.*np.pi*freqs[k]*t + phis[k])
    return y,t

def synth_signal(sigtype='sin', freq=200.0, amp=1.0, phi=0.0, sample_rate=8000., dur=0.25):
    '''
    synthesizes a short segment of a (semi) harmonic signal
    
    This is mainly a wrapper around scipy.signal with physical units for sample rate, frequency and duration
    
    Recognized signal types are:
        sin, square, sawtooth, triangle, pulsetrain, 
        chirp 1:20, chirp 20:1, gausspulse, "modulated white noise", "Dual Tone"
        
        fourier
        
    Input Paramters:
    ----------------
    sigtype :     str, default 'sin'
        signal type, mostly similar to naming in scipy.signal
    freq :        float, default 200
        fundamental frequency in Hz
    amp :
        amplitude
    phi :
        phase in radials
    sample rate :  float (default=8000)
        sampling rate in Hz
    dur :          float (default =0.250)
        duration in seconds

    Returns:
    --------
        s     np.array of floats
            synthesized signal
        t     np.array of floats
            sample times (in seconds, starting at 0.0)    
            
    '''
    t = np.linspace(0.0, dur, int(dur*sample_rate), endpoint=False)
    npts = len(t)
    tt = 2.*np.pi*(freq*t) + phi
    if sigtype == 'sin':
        y = np.sin(tt)
    elif sigtype == 'fourier':
        y,t = synth_fourier(freqs=freq,amps=amp,phis=phi,dur=dur,sample_rate=sample_rate)
        amp = 1.0 # no further scaling needed with global amplitude
    elif sigtype == 'square':
        y = signal.square(tt)
    elif sigtype == 'sawtooth':
        y = signal.sawtooth(tt)
    elif sigtype == 'triangle':
        y = signal.sawtooth(tt,width=.5)
    elif sigtype == 'pulsetrain':
        y = np.zeros(t.shape)
        nperiod = int((1./freq)*sample_rate)
        for k in range(5,npts,nperiod):
            y[k] = 1.
    elif sigtype == 'chirp 1:20':
        y = signal.chirp(t,freq,dur,20.*freq,method='linear')
    elif sigtype == 'chirp 20:1':
        y = signal.chirp(t,20*freq,dur,freq,method='linear')
    elif sigtype == 'gausspulse':
        y = signal.gausspulse(t-dur/2.,fc=freq,bw=.4)
    elif sigtype == 'white noise':
        y = np.random.randn(len(t))/4.
    elif sigtype == 'modulated white noise':
        y = np.random.randn(len(t))*(np.sin(tt)+1.)/8.
    elif sigtype == 'Dual Tone':   #DTMF tone ratios is approximately 21/19 ~ 1.1054 (697,770,852,941)*(1209,1336,1477)
                              # row/col ratio is 1.735
        tt1 = 2.*np.pi*(1.735*freq*t) + phi
        y = 0.5*np.sin(tt) + 0.5*np.sin(tt1) 

    else:
        print( 'signal: Unrecognized signal type')
    return amp*y, t


def synth_griffinlim(S,sample_rate=8000,shift=0.01,mode='dB'):
    '''
    Griffin Lim Synthesis
    
    Input:
    ------
        S:           Spectrogram (n_coeff,n_frames)
        sample_rate: sampling rate (default=8000)
        shift:       frame shift in seconds (default=0.01)
        mode:        power mode of spectrogram (default='dB')
        
    Returns:
    --------
        y:           synthesized signal
    '''
    
    # convert input spectrogram to magnitude spectrogram
    S_mag = set_mode(S,mode,'magnitude')
    hop_length = int(shift*sample_rate)
    y = librosa.griffinlim(S_mag,hop_length=hop_length)
    return(y)

