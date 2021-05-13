import math
import numpy as np
import librosa
from spchutils.constants import EPS_FLOAT

def spectrogram(y,sample_rate=16000,frame_shift=10.,frame_length=30.,preemp=0.97,n_fft=512,window='hamm',output='dB',n_mels=None):
    '''
    spectrogram is a wrapper making use of the librosa() stft library 
    + with arguments for frame_shift and frame_length in msec 
    + and with  some adjustments on frame positioning similar to Kaldi / SPRAAK
        - (a) frame positioning : centered at: k*n_shift + n_shift/2
        - (b) #frames:  n_samples // n_shift (first and last frame partially filled with mirrored data)
        - (c) edge processing: mirroring of input signal around the edges
        - (d) pre-emphasis is applied after edge processing  (librosa does it before)

    required arguments:
      y       waveform data (numpy array) 

    optional arguments:
      sample_rate  sample rate in Hz, default=16000
      frame_shift  frame shift in msecs, default= 10.0 msecs
      frame_length frame length in msecs, default= 30.0 msecs
      preemp       preemphasis coefficient, default=0.95
      window       window type, default='hamm'
      n_fft        number of fft coefficients, default=512
      n_mels       number of mel coefficients, default=None
      output       output scale, default='dB', options['dB','power']   

    output:
      spectrogram  numpy array of size [nfreq,nframes] ; values are in dB
         
    '''
    
    n_shift = int(float(sample_rate)*frame_shift/1000.0)
    n_length = int(float(sample_rate)*frame_length/1000.0)
    if n_fft < n_length :
        print('Warning(Spectrogram): n_fft raised to %d'%n_length)
        n_fft = n_length
    
    # extend the edges by mirroring
    ii = n_shift//2
    n_pad = n_fft//2
    z=np.concatenate((y[0:n_pad][::-1],y,y[:-n_pad-1:-1]))
    z[0]=(1.-preemp)*y[0]
    z[1:]= z[1:] - preemp*z[0:-1]
    y_pre = z[ii:len(z)-ii]
   
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
# for synchronization between different shifts, we place points at
#     ti = (i+offs) * dt    with offs=0.5 by default
# inputs can be scalars, lists or numpy arrays  outputs are always numpy arrays
def t2indx(t,dt=1.,align='center'):
    """ time-to-index conversion:  ; see indx2t() for details"""
    offs = 0.5 if align=='center' else 0.0
    return np.round((np.array(t).astype(float)/float(dt)-offs)).astype(int)
def indx2t(i,dt=1.,align='center'):
    """ index-to-time conversion: 
        time[i] = (i+offs) * dt  ; offs=0.5 when 'center'(default)
        
    dt : sampling period
    align : default(='center') """
    offs = 0.5 if align=='center' else 0.0
    return (np.array(i).astype(float) + offs )*dt
def time_range(n,dt=1.,align='center'):
    """ indx2t() for n samples 0 ... n-1 """
    offs = 0.5 if align=='center' else 0.0
    return (np.arange(n,dtype='float32')+offs)*dt 

