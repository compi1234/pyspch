# A set of utilities for audio IO and essential handling
#
# Data Formats: 
# audio data is passed as (float32) numpy arrays using the 'implicit mono' convention ( librosa) 
#    + (n_samples,_) for mono 
#    + (n_channels,n_samples) for multichannel data 
#
# Alternatives:
# (i) PyTorchAudio uses Explicit mono with channel first: i.e.
#    all data is stored in 2D arrays (n_channels,n_samples) 
# 
# (ii) Audio Streaming Formats 
#     always organize data as  n_samples * ( n_channels (* sample_width) )
#     This is more convenient for I/O, but less convenient for speech processing
#
# Dependencies: 
#   - soundfile >= 0.9 
#   - pydub
#   - librosa = 0.8.0
#   - numba>= 0.43.0,<= 0.48.0
#   - (Google Colab): soundfile and pydub may need to be installed before you can use this
#
# Credits:
# https://gist.github.com/korakot/record.py  for the javascript recording code
#
import os,sys,io 
from urllib.request import urlopen
from urllib.parse import urlparse
from IPython.display import display, Audio,  Javascript
import IPython   
import math
import numpy as np
from base64 import b64decode

import librosa

# we currently acknowledge 2 IO METHODS, 
# 1. "sd" : use the sounddevice on the local machine
#   -> do I/O via sounddevice
# 2. "colab" : use javascript in the browser for audio I/O when server based as in Google COLAB
#   -> recording is done via javascript in Browser with google.colab.output() for
#          returning the data 
#   -> playing: 
#          an Ipython.display.Audio() is generated
#

_IO_ENV_ = None
if ('google.colab' in str(get_ipython())):
    _IO_ENV_ = "colab"
    from google.colab import output
else:
    try:
        import sounddevice as sd
        _IO_ENV_ = "sd"
    except:
        print("ERROR(audio): sounddevice module not found, pls. install if you need local audio\n  Alternatively you can run your notebook in colab")
        
try:    
    import soundfile as sf
    from pydub import AudioSegment 
except:
    print("ERROR: Using spchutils.audio requires soundfile and pydub packages to be installed.   You should fix this first")


def get_fobj(resource):
    '''
    returns a file like object for reading from either a filename (local) or a URL resource
    
    A URL resource is read into a BytesIO object, while filenames are left unchanged
    '''
    parsed=urlparse(resource)
    if(parsed.scheme !=''):
        fobj = io.BytesIO(urlopen(resource).read())
    else:
        fobj = resource
    return(fobj)

def load(resource,sample_rate=None,**kwargs):
    ''' 
    This is a tiny wrapper around librosa.load() to accomodate for specifying a resource
    both by url or filename
    
    Parameters:
    -----------
        resource : string          
            url or file name
        sample_rate : int (optional)
            if given, resample to the target sampling rate
        **kwargs 
            extra parameters to be passed to librosa
            e.g. mono(boolean)
            
    Returns:
    --------
        wavdata : float-32 array, shape (n_samples, ) for mono or (n_channels, n_samples)
            the waveform data are scaled to [-1., 1.]
        sample_rate : int
            sampling rate of returned signal
    '''
    
    fobj = get_fobj(resource)
    data, sample_rate = librosa.load(fobj,dtype='float32',sr=sample_rate,**kwargs)
    # sample rate conversion may result in values exceeding +-1, so a little bit of clipping
    # can resolve this 
    return(np.clip(data,-1.,1.),sample_rate)

    # similar code with soundfile (without sample rate conversion as in librosa)
    # src_data, src_sample_rate = sf.read(fp,dtype='float32',always_2d=always_2d,**kwargs)


def save(filename,wavdata,sample_rate,**kwargs):
    """ Save a one or multi-d waveform data using soundfile """
    if wavdata.ndim == 1:
        sf.write(filename, data, sample_rate, **kwargs)
    else:
        sf.write(filename, data.T, sample_rate, **kwargs)        

    
def play(wavdata,sample_rate=16000, channels=None, wait=False):
    """
    Play an audio waveform either via the local sounddevice or display a HTML/js object
    
    This routine checks the global variable _IO_ENV_, which is defined on loading
    the audio module, to know where to put output / get input    

    
    Parameters
    ----------
        wavdata : array of (n_channels,n_samples) or (n_samples,)
            waveform data
        sample_rate : int
            sampling rate (default=16000)
        channels : array of int's (default = None)
            channels to be played, if None all channels are played 
        wait : boolean (default=False)
            wait to return till play is finished (only applicable to sounddevice)


"""
    global _IO_ENV_
    
    if wavdata.ndim == 1:  # you only have mono, reshape to 2D and neglect the channels argument    
        play_data = wavdata.reshape(1,-1)
        n_channels = 1
    else:
        if channels == None:
            channels = np.arange(wavdata.shape[0])          
        n_channels = len(channels)        
        play_data = wavdata[channels]
        
    # render audio directly on device or via IPython.display.Audio object
    if _IO_ENV_ == 'sd':
        sd.play(play_data.T,sample_rate)
        if(wait): sd.wait()
    else:   
        if IPython.version_info[0] >= 6:
            kwargs = {'normalize':False}
        else:
            print("Warning: you are using IPython<6 which will auto-normalize audio output")
            kwargs = {}
        if n_channels == 1:
            display(Audio(data=play_data[0],rate=sample_rate,**kwargs))
        elif n_channels == 2:
            display(Audio(data=(play_data[0],play_data[1]),rate=sample_rate,**kwargs))
        else:
            print("Warning(play): Too many channels requested, I will play the first channel only")
            display(Audio(data=play_data[0],rate=sample_rate))


  
    
def record(seconds=2.,sample_rate=16000,n_channels=1):
    """
    This routine checks the global variable _IO_ENV_, which is defined on loading
    the audio module, to know where to put output / get input  
    
    Parameters
    ----------
        seconds : float
            number of seconds to record (default=2.0)
        sample_rate : int
            sampling rate (default=16000)
        n_channels : int
            number of channels to record (default=1)

            
    Returns
    -------
        wavdata : float-32 array, shape (n_sample,_) or (n_channels, n_samples)
            the waveform data scaled to [-1., 1.]

    """
    if _IO_ENV_ =='sd':
        data = _record_sd(seconds,sample_rate,n_channels=n_channels)
    elif _IO_ENV_ == 'colab':
        data = _record_colab(seconds,sample_rate,n_channels = n_channels)
    else:
        print("ERROR(record): unknown _IO_ENV_ to record from ")
    # return 1D data for mono, 
    data = data.T
    if(data.shape[0]==1): return(data.ravel())
    else: return(data)
    
# record using sounddevice
def _record_sd(seconds,sample_rate,n_channels=1):
        data = sd.rec(int(seconds*sample_rate),samplerate=sample_rate,channels=n_channels)
        print('recording started for %.2f seconds on %d channel(s)' % (seconds,n_channels) )
        sd.wait()  # waits for recording to complete, otherwise you get nonsense
        print('recording finished')
        return(data) 
        

# need to insert multichannel request here somehow
# see e.g. https://w3c.github.io/mediacapture-main/getusermedia.html#mediastreamconstraints
#    audio: {
#      deviceId: localStorage.micId,
#      channelCount: 2
#    }

_RECORD_JS = """
const sleep  = time => new Promise(resolve => setTimeout(resolve, time))
const b2text = blob => new Promise(resolve => {
  const reader = new FileReader()
  reader.onloadend = e => resolve(e.srcElement.result)
  reader.readAsDataURL(blob)
})

var record = time => new Promise(async resolve => {
  stream = await navigator.mediaDevices.getUserMedia({ audio: true })
  recorder = new MediaRecorder(stream)
  chunks = []
  recorder.ondataavailable = e => chunks.push(e.data)
  recorder.start()
  await sleep(time)
  recorder.onstop = async ()=>{
    blob = new Blob(chunks)
    text = await b2text(blob)
    resolve(text)
  }
  recorder.stop()
})
"""

def _record_colab(seconds,sample_rate, n_channels: int = None,): 
#  Based on: https://gist.github.com/korakot/c21c3476c024ad6d56d5f48b0bca92be
#  and on: https://github.com/magenta/ddsp/blob/master/ddsp/colab/colab_utils.py
      """Record using JavaScript in the browser 
         and convert returned audio (in bytes) into a float32 numpy array using Pydub.

      Args:
        secs: seconds to record
        sample_rate: Resample recorded audio to this sample rate.
        n_channels: If not specified, output shape will be based on the contents
          of wav_data. Otherwise, will force to be 1 or 2 channels.
        normalize_db: Normalize the audio to this many decibels. Set to None to skip
          normalization step.

      Returns:
        An array of the recorded audio at sample_rate. If mono, will be shape
        [samples], otherwise [channels, samples].
      """

      print('Starting recording for {} seconds...'.format(seconds))
      display(Javascript(_RECORD_JS))
      # using Google Colab's eval_js to return the data !!
      s = output.eval_js('record(%d)' % (seconds*1000.))
      print('Finished recording!')
      audio_bytes = b64decode(s.split(',')[1])

      # Parse and normalize the audio.
      aseg = AudioSegment.from_file(io.BytesIO(audio_bytes))
      # Convert to the correct sampling rate
      aseg = aseg.set_frame_rate(sample_rate)
    
      # Convert to numpy array.
      channel_asegs = aseg.split_to_mono()
      if n_channels:
        aseg = aseg.set_channels(n_channels)
      samples = [s.get_array_of_samples() for s in channel_asegs]
      fp_arr = np.array(samples).astype(np.float32)
      fp_arr /= np.iinfo(samples[0].typecode).max

      # If only 1 channel, remove extra dim.
      if fp_arr.shape[0] == 1:
        fp_arr = fp_arr[0]

      return fp_arr

