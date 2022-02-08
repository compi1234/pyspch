import math
import numpy as np
import librosa
from ..utils.constants import EPS_FLOAT


# do preemhasis and padding
def pad_and_preemp_(y,pad=None,preemp=None):
    '''
    constructs an array that is padded and preemphasized
    to make the output suitable for librosa processing with center = False (with a predictable number of frames)
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
    y_pre = pad_and_preemp_(y,pad=pad,preemp=preemp)
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

def spectrogram(y,sample_rate=16000,f_shift=0.01,f_length=0.03,preemp=0.97,window='hamm', n_mels=None,output='dB'):
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
      output       output scale, default='dB', options['dB','power']   

    output:
      spectrogram  numpy array of size [nfreq,nframes] ; values are in dB
    '''
    n_shift = int(float(sample_rate)*f_shift)
    n_length = int(float(sample_rate)*f_length)
    n_fft = 2**math.ceil(math.log2(n_length))
    pad = (n_fft-n_shift)//2
    y_pre = pad_and_preemp_(y,pad=pad,preemp=preemp)
    spg_stft = librosa.stft(y_pre,n_fft=n_fft,hop_length=n_shift,win_length=n_length,window=window,center=False)
    spg_power = np.abs(spg_stft)**2
    if n_mels is None:
        spg = spg_power
    else:
        spg = librosa.feature.melspectrogram(S=spg_power,n_mels=n_mels,sr=sample_rate) 

    if output == 'dB':    return(librosa.power_to_db(spg,amin=EPS_FLOAT))
    else:                 return(spg)
    
def spg2mel(spg,n_mels=80,sample_rate=16000):
    '''
    convert standard spectrogram (in dB) to mel spectrogram (in dB)

    '''
    mel_power = librosa.feature.melspectrogram(
                    S=librosa.db_to_power(spg),n_mels=n_mels,sr=sample_rate
                    )
    return(librosa.power_to_db(mel_power,amin=EPS_FLOAT)) 


def spectrogram_old(y,sample_rate=16000,f_shift=0.01,f_length=0.03,preemp=0.97,n_fft=None,window='hamm',output='dB',n_mels=None):
    '''
    This Version will be deprecated soon ..
    
    spectrogram is a wrapper making use of the librosa() stft library 
    + with arguments for f_shift and f_length in secs 
    + and with  some adjustments on frame positioning similar to Kaldi / SPRAAK
        - (a) frame positioning : centered at: k*n_shift + n_shift/2
        - (b) #frames:  n_samples // n_shift (first and last frame partially filled with mirrored data)
        - (c) edge processing: mirroring of input signal around the edges
        - (d) pre-emphasis is applied after edge processing  (librosa does it before)

    required arguments:
      y       waveform data (numpy array) 

    optional arguments:
      sample_rate  sample rate in Hz, default=16000
      f_shift      frame shift in secs, default= 0.010 secs
      f_length     frame length in secs, default= 0.030 secs
      preemp       preemphasis coefficient, default=0.95
      window       window type, default='hamm'
      n_fft        number of fft coefficients, default=None, i.e. smallest power of 2 larger than n_length
      n_mels       number of mel coefficients, default=None
      output       output scale, default='dB', options['dB','power']   

    output:
      spectrogram  numpy array of size [nfreq,nframes] ; values are in dB
         
    '''
    
    n_shift = int(float(sample_rate)*f_shift)
    n_length = int(float(sample_rate)*f_length)
    if n_fft is None:
        n_fft = 2**math.ceil(math.log2(n_length))
    if n_fft < n_length :
        print('Warning(Spectrogram): n_fft raised to %d'%n_length)
        n_fft = n_length
    
    # extend the edges by mirroring
    ii = n_shift//2
    n_pad = n_fft//2
    z=np.concatenate((y[0:n_pad][::-1],y,y[:-n_pad-1:-1]))
    z[0]=(1.-preemp)*z[0]
    z[1:]= z[1:] - preemp*z[0:-1]
    y_pre = z[ii:len(z)-ii]
   
    print('specg',len(y_pre))
    spg_stft = librosa.stft(y_pre,n_fft=n_fft,hop_length=n_shift,win_length=n_length,window=window,center=False)
    spg_power = np.abs(spg_stft)**2
    
    if n_mels is None:   spg = spg_power
    else:                spg = librosa.feature.melspectrogram(S=spg_power,n_mels=n_mels,sr=sample_rate) 
        
    if output == 'dB':    return(librosa.power_to_db(spg,amin=EPS_FLOAT))
    else:                 return(spg)


           
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

