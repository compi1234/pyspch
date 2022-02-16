""" Utilities for Spectral Processing """

import math
import numpy as np
import librosa
from scipy.fft import dct, idct
DCT_TYPE = 2  # dct_type used to convert log-spectrum to cepstrum
DCT_NORM = 'ortho'
 
from ..utils.constants import EPS_FLOAT, LOG2DB
from .frames import *

def check_mode(mode):
    if mode not in ('magnitude','power','dB'):
        raise ValueError(f"Unsupported mode: {mode} select one from 'magnitude', 'power', 'dB' ")

def set_mode(S,mode_in,mode_out):
    '''
    converts (spectrograms) between 'magnitude','power' and 'dB'
     '''
    
    check_mode(mode_in)
    check_mode(mode_out)
    if mode_in == mode_out: return(S)
    elif mode_in == 'power':
        if mode_out == 'dB':
            return(librosa.power_to_db(S,amin=EPS_FLOAT))
        elif mode_out == 'magnitude':
            return( S ** 0.5 )
    elif mode_in == 'magnitude':
        if mode_out == 'power':
            return( S ** 2.0 )
        elif mode_out == 'dB':
            return(librosa.power_to_db(S ** 2.0,amin=EPS_FLOAT))
    elif mode_in == 'dB':
        if mode_out == 'power':
            return(librosa.db_to_power(S))
        elif mode_out == 'magnitude':
            return( librosa.db_to_power(S) ** 0.5 )

    
def spectrogram(y,sample_rate=16000,f_shift=0.01,f_length=0.03,preemp=0.97,window='hamm', n_mels=None,mode='dB'):
    '''
    spectrogram is a wrapper making use of the librosa() stft library 
    + with arguments for f_shift and f_length in secs 
    + and with  some adjustments on frame positioning similar to Kaldi / SPRAAK
        - (a) frame positioning : centered at: k*n_shift + n_shift/2
        - (b) #frames:  n_samples // n_shift (first and last frame partially filled with mirrored data)
        - (c) edge processing: mirroring of input signal around the edges
        - (d) pre-emphasis is applied after edge processing  (librosa does it before)
    + FFT window size is set to smallest power of 2 >= window length in samples
    + optionally a mel-spectrogram is computed

    required arguments:
      y       waveform data (numpy array) 

    optional arguments:
      sample_rate  sample rate in Hz, default=16000
      f_shift      frame shift in secs, default= 0.010 secs
      f_length     frame length in secs, default= 0.030 secs
      preemp       preemphasis coefficient, default=0.97
      window       window type, default='hamm'
      n_mels       number of mel coefficients, default=None
      mode         mode of output, default='power', options['dB','power']   

    output:
      spectrogram  numpy array of size [n_freq,n_frames] 
    '''
    n_shift = int(float(sample_rate)*f_shift)
    n_length = int(float(sample_rate)*f_length)
    n_fft = 2**math.ceil(math.log2(n_length))
    pad = (n_fft-n_shift)//2
    y_pre = preemp_pad(y,pad=pad,preemp=preemp)
    spg_stft = librosa.stft(y_pre,n_fft=n_fft,hop_length=n_shift,win_length=n_length,window=window,center=False)
    spg_power = np.abs(spg_stft)**2
    if n_mels is None:
        spg = spg_power
    else:
        spg = librosa.feature.melspectrogram(S=spg_power,n_mels=n_mels,sr=sample_rate) 

    return(set_mode(spg,'power',mode))
    
##############################

def spg2mel(S,n_mels=80,sample_rate=16000,fmin=40.,fmax=None,mode='dB'):
    '''
    convert standard (power) spectrogram to mel (power) spectrogram
    mode is selectable
    
    fmin   lowest edge of lowest filterband, defaults to 40Hz
    fmax   highest edge of highest filterband, if None, defaults to 0.45*sr

    The librosa fmin (0Hz), fmax (0.5*sr) defaults favor a 'wide' frequency range, suitable for HIFI recordings but susceptible to low frequency humm and high frequency anti-aliasing effects
    Therefore our defaults are just a bit more conservative.
    '''

    S = set_mode(S,mode,'power')
    if fmax is None : fmax = 0.45*sample_rate 
    S_mel = librosa.feature.melspectrogram(
         S=S,n_mels=n_mels,sr=sample_rate,fmin=fmin,fmax=fmax,
                    )
    return(set_mode(S_mel,'power',mode))


def cepstrum(y=None,S=None,n_cep=None,sample_rate=16000,mode='dB'):
    """
    cepstrum returns the non truncated cepstrum if n_cep is None, 
        otherwise the  cepstrum truncated to n_cep coeffiecients
        
    note1: we are using scipy DCT Type II with Ortho normalization, 
        same as in pytorch.get_dct_matrix()
    note: we compute idct of the log-spectrum (as in HTK, KALDI) not of the db_spectrum (as in librosa)
    
       
    Arguments:
    y :       waveform to compute cepstrum from, if None use spg
    S :       log or power-spectrogram nd.array of size (n_param,n_fr)
    n_cep :   number of cepstral coefficients to return (default = None = all)
    mode :    'dB'(default) or 'power'
    
    Returns:
    cep :     cepstrogram of size (n_cep,n_fr)
    """

    if y is not None: # compute from waveform
        mode = 'dB'
        S = spectrogram(y,sample_rate=sample_rate,n_mels=None,mode=mode)
    else: # compute from spectrogram
        if S is None:
            raise ParameterError("Either `y` or `S` must be input.")
    spg = set_mode(S,mode,'dB')
    cep = dct(spg/LOG2DB,axis=0,type=DCT_TYPE,norm=DCT_NORM)
    if n_cep is None:    return(cep)
    else: 
        return(cep[0:n_cep,:])



def melcepstrum(y=None,S=None,n_cep=13,n_mels=80,sample_rate=16000,fmin=40.,fmax=None,mode='dB'):
    """
    cepstrum returns the non truncated cepstrum if n_cep is None, 
        otherwise the truncated cepstrum
    
    Arguments:
    y :       waveform to compute mel cepstrum from, if None use spg
                default spectrogram parameters will be used 
    spg :     log-spectrogram nd.array of size (n_param,n_fr), if None y must be specified
    n_cep :   number of cepstral coefficients to return (default = 13)
    
    Returns:
    cep :     cepstrogram of size (n_cep,n_fr)
    """
               
    if y is not None:
        mode = 'dB'
        spgmel = spectrogram(y,sample_rate=sample_rate,n_mels=n_mels,mode=mode)
    else:
        if S is None:
            raise ParameterError("Either `y` or `S` must be input.")
        if fmax is None : fmax = 0.45*sample_rate 
        spgmel = spg2mel(S,sample_rate=sample_rate,n_mels=n_mels,fmin=fmin,fmax=fmax,mode=mode)
        
    spgmel = set_mode(spgmel,mode,'dB')
    cep = dct(spgmel/LOG2DB,axis=0,type=DCT_TYPE,norm=DCT_NORM)
               
    if n_cep is None:    return(cep)
    else: return(cep[0:n_cep,:])
    
# returns evelope and residue spectra from cepstrum
# cep will first be padded to size 'n_spec' before applying the (i)dct
def cep_lifter(cep,n_lifter=None,n_spec=128):
    '''
    cep_lifter computes the spectral envelope and/or residue from the cepstrum
        1. first the cep input is zero padded to n_spec size
        2. the 0-padded is split in an envelope and residue part, as specified by n_lifter
        3. A a DCT is performed to both components
        
    If no lifter is specified then just the cepstral envelope is returned
    
    returns log-spectra of envelope and residue
    '''
    
    n_cep, n_frames = cep.shape
    if n_cep > n_spec : # unusual
        cep = cep[0:n_spec,:]
        n_cep = n_spec
    if n_lifter is not None:
        n_cep = n_lifter
    
    # extend cepstrum to n_spec coefficients 
    cep = np.vstack( (cep, np.zeros((n_spec-n_cep,n_frames))) )
    
    # create the envelope part
    cep_env = np.vstack( (cep[0:n_cep,:], np.zeros((n_spec-n_cep,n_frames)) ))
    spec_env = LOG2DB*idct(cep_env,axis=0,type=DCT_TYPE,norm='ortho')
        
    if n_lifter is None:
        return(spec_env)
    else:
        # create the residue matrix
        cep_res = np.vstack( (np.zeros((n_cep,n_frames)), cep[n_cep:n_spec,:] ))
        spec_res = LOG2DB*idct(cep_res,axis=0,type=DCT_TYPE,norm='ortho')

        return(spec_env,spec_res)

           

