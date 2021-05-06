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
#   - soundfile 
#   - pydub
#   - librosa = 0.8.0
#   - (Google Colab): soundfile and pydub may need to be installed before you can use this
#
import os,sys,io 
import scipy.signal

from urllib.request import urlopen
from urllib.parse import urlparse
from IPython.display import display, Audio, HTML, Javascript
import soundfile as sf    
import math
import numpy as np
from base64 import b64decode
import matplotlib.pyplot as plt

import librosa

# we acknowledge 2 IO modes, specified with IO_DEVICE
# 1. "browser" : use browser for audio I/O when server based as in Google COLAB
#   -> recording is done via javascript in Browser
#   -> playing is done via the HTML based display.Audio in Ipython
# 2. "sd" : use the sounddevice on the local machine
#   -> do I/O via sounddevice
#
try:
    import google.colab
    IN_COLAB = True
    IO_DEVICE = "browser"
    from google.colab import output
except:
    IN_COLAB = False
    IO_DEVICE = "sd"
try:
    import sounddevice as sd
except:
    if IN_COLAB == False:
        print("sounddevice module not found, pls. install if you need local audio")
    IO_DEVICE = "browser"
import soundfile as sf
from pydub import AudioSegment 


def get_fobj(resource):
    '''
    get_fobj() returns a file like object for reading
    i.e. a URL resource is put into a BytesIO object
    while filenames are left unchanged
    '''
    parsed=urlparse(resource)
    if(parsed.scheme !=''):
        fobj = io.BytesIO(urlopen(resource).read())
    else:
        fobj = resource
    return(fobj)


def load(resource,sample_rate=None,**kwargs):
    ''' 
    load() is a tiny wrapper around soundfile.read() to accomodate for
    both url and filename arguments
    
    Parameters:
    -----------
        resource : string          
            url or file name
        sample_rate : int (optional)
            if given, resample to the target sampling rate
        **kwargs 
            extra parameters to be passed to librosa
            mono(boolean), sr(resampling rate)
            
    Returns:
    --------
        wavdata : float-32 array, shape (n_samples, ) for mono or (n_channels, n_samples)
            the waveform data are scaled to [-1., 1.]
        sample_rate : int
            sampling rate of returned signal
    '''
    
    fobj = get_fobj(resource)
    data, sample_rate = librosa.load(fobj,dtype='float32',sr=sample_rate,**kwargs)
    return(data,sample_rate)

    # similar code with soundfile
    # src_data, src_sample_rate = sf.read(fp,dtype='float32',always_2d=always_2d,**kwargs)


def save(filename,wavdata,sample_rate,**kwargs):
    if wavdata.ndim == 1:
        sf.write(filename, data, sample_rate, **kwargs)
    else:
        sf.write(filename, data.T, sample_rate, **kwargs)        
    print("audio.save() not implemented Yet")

    
def play(wavdata,sample_rate=16000, channels=None, io_device=IO_DEVICE, wait=False):
    """
    Play and audio waveform either the local sounddevice or display a browser object
    
    Parameters
    ----------
        wavdata : array of (n_channels,n_samples)
            waveform data
        sample_rate : int
            sampling rate (default=16000)
        channels : array of int's (default = None)
            channels to be played, if None all channels are played 
        io_device : string
            device from which to record (default='sd')
            currently supporting 'sd' or 'browser'
        wait : boolean (default=False)
            wait to return till play is finished (only applicable to sounddevice)
    """


    if wavdata.ndim == 1:  # you only have mono, neglect the channels argument    
        play_data = wavdata
        n_channels = 1
    else:
        if channels == None:
            channels = np.arange(wavdata.shape[0])          
        n_channels = len(channels)        
        play_data = wavdata[channels]
        
    # use IPython display.Audio
    if io_device == 'browser':
        if n_channels == 1:
            display(Audio(data=play_data[0],rate=sample_rate))
        elif n_channels == 2:
            display(Audio(data=(play_data[0],play_data[1]),rate=sample_rate))
        else:
            print("Warning(play): Too many channels requested, I will play the first channel only")
            display(Audio(data=play_data[0],rate=sample_rate))
    elif io_device == 'sd':
        sd.play(play_data.T,sample_rate)
        if(wait): sd.wait()
    else:
        print('No known method/device to play sound')

  
    
def record(seconds=2.,sample_rate=16000,n_channels=1, io_device = IO_DEVICE ):
    """
    Parameters
    ----------
        seconds : float
            number of seconds to record (default=2.0)
        sample_rate : int
            sampling rate (default=16000)
        n_channels : int
            number of channels to record (default=1)
        io_device : string
            device from which to record (default='sd')
            currently supporting 'sd' or 'browser'
            
    Returns
    -------
        wavdata : float-32 array, shape (n_sample,_) or (n_channels, n_samples)
            the waveform data scaled to [-1., 1.]

    """
    if io_device =='sd':
        data = _record_sd(seconds,sample_rate,n_channels=n_channels)
    elif io_device == 'browser':
        data = _record_js(seconds,sample_rate,num_channels = n_channels)
    
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
        

RECORD = """
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

def _record_js(seconds,sample_rate, num_channels: int = None,): 
#  Based on: https://gist.github.com/korakot/c21c3476c024ad6d56d5f48b0bca92be
#  and on: https://github.com/magenta/ddsp/blob/master/ddsp/colab/colab_utils.py
      """Record via the browser using JavaScript
         and convert returned audio (in bytes) into a float32 numpy array using Pydub.

      Args:
        secs: seconds to record
        sample_rate: Resample recorded audio to this sample rate.
        num_channels: If not specified, output shape will be based on the contents
          of wav_data. Otherwise, will force to be 1 or 2 channels.
        normalize_db: Normalize the audio to this many decibels. Set to None to skip
          normalization step.

      Returns:
        An array of the recorded audio at sample_rate. If mono, will be shape
        [samples], otherwise [channels, samples].
      """

      print('Starting recording for {} seconds...'.format(seconds))
      display(Javascript(RECORD))
      s = output.eval_js('record(%d)' % (seconds*1000.))
      print('Finished recording!')
      audio_bytes = b64decode(s.split(',')[1])

      # Parse and normalize the audio.
      aseg = AudioSegment.from_file(io.BytesIO(audio_bytes))
      # Convert to the correct sampling rate
      aseg = aseg.set_frame_rate(sample_rate)
    
      # Convert to numpy array.
      channel_asegs = aseg.split_to_mono()
      if num_channels:
        aseg = aseg.set_channels(num_channels)
      samples = [s.get_array_of_samples() for s in channel_asegs]
      fp_arr = np.array(samples).astype(np.float32)
      fp_arr /= np.iinfo(samples[0].typecode).max

      # If only 1 channel, remove extra dim.
      if fp_arr.shape[0] == 1:
        fp_arr = fp_arr[0]

      return fp_arr




############################################################# 
# Plotting Routines have moved to spectrogram module !!
# 
##############################################################

def make_row_grid(height_ratios=[1.,3.],figsize=(10,4)):
    """
    Setup a figure for a multi-row plot
    
    """
    fig = plt.figure(figsize=figsize,clear=True,constrained_layout=True)
    nrows = len(height_ratios)
    gs = fig.add_gridspec(nrows=nrows,ncols=1,height_ratios=height_ratios)
    ax = []
    for i in range(0,nrows):
        axx = fig.add_subplot(gs[i,0])
        ax.append(axx)
    return(fig,ax)

def add_line_plot(ax,y,x=None,xscale=1.,ylim=None,xlabel=None,ylabel=None,**kwargs):
    """
    Add a line plot to an existing axis
    x and y are (1-D) series
    to scale to seconds, use: xscale=sample_rate
    """
    if x==None:
        x = np.arange(len(y)) / xscale
    ax.plot(x,y,**kwargs)
    ax.set_xlim(0,x[-1])
    if ylim != None:  ax.set_ylim(ylim)
    if xlabel != None: ax.set_xlabel(xlabel)
    if ylabel != None: ax.set_ylabel(ylabel)
        
def _old_add_img_plot(ax,img,xscale=1.,yscale=1.,xlabel=True):
    """
    Add an image plot to a given axis with typical spectrogram layout
    """
    (nr,nc)= img.shape
    print(nr,nc)
    extent = [-0.5, (float(nc)-.5)/xscale, -.5, (float(nr)-.5)/yscale]
    ax.imshow(img,cmap='jet',aspect='auto',origin='lower',extent=extent)

    if(xlabel):
        ax.tick_params(axis='x',labelbottom=True)
    else:
        ax.tick_params(axis='x',labelrotation=0.0,labelbottom=False,bottom=True)     

def add_img_plot(ax,img,xscale=1.,yscale=1.,xticks=True,xlabel=None,ylabel=None,**kwargs):
    '''
    Add an image plot (spectrogram style)
    
    Parameters:
    -----------
    ax :     axis
    img :    image
    xscale : (float) - scale to apply to x-axis (default=1.)
    yscale : (float) - scale to apply to y-axis (default=1.)
    xticks : (boolean) - label the x-axis ticks
    **kwargs: extra arguments to pass / override defaults in plt.imshow()
    '''
    
    (nr,nc)= img.shape
    extent = [-0.5, (float(nc)-.5)/xscale, -.5, (float(nr)-.5)/yscale]
    
    xargs={'cmap':'jet','origin':'lower','aspect':'auto'}
    xargs.update(kwargs)
    ax.imshow(img,extent=extent,**xargs)
    
    if(xticks): ax.tick_params(axis='x',labelbottom=True)
    else:       ax.tick_params(axis='x',labelrotation=0.0,labelbottom=False,bottom=True)         
    if xlabel != None: ax.set_xlabel(xlabel)
    if ylabel != None: ax.set_ylabel(ylabel)
        
# xscale here is a hack, as we really should pass xlim
def _old_add_seg_plot(ax,df,xscale=1.,yscale=1.):
    """
    Add a segmental plot
    """
    print("waiting for implementation")
    plot_seg(ax,df,xlim=[0.,xscale])

def add_seg_plot(ax,df,**kwargs):
    
    ''' 
    add_seg_plot(): adds a segmentation to an axis

    Parameters:
    -----------
    ax:         matplotlib axis
    df:         dataframe

    **kwargs:
    xlim:       X-axis range (default: [0 1])
    ytxt        height at which to write out the segmentation (default= 0.5)
    Vlines      flag for plotting segmentation lines (default=True)
    linestyle   default='solid'
    linecolor   default='k'
    fontsize    default=14
    ''' 

    xargs={'xlim':[0.,1.],'ytxt':0.5,'fontsize':14,'Vlines':True,'linestyle':'solid', linecolor:'k'}
    xargs.update(kwargs)
    
    # First plot a dummy axis to avoid matplotlib going wild
    ax.imshow(np.zeros((1,1)),aspect='auto',cmap='Greys',vmin=0.,vmax=1) 
    for iseg in range(0,len(df)):
        i1= df['t0'][iseg]
        i2= df['t1'][iseg]
        txt = df['seg'][iseg]
        if(xargs['Vlines']):
            ax.vlines([i1,i2],0.,1.,linestyles=xargs['linestyle'],colors=xargs['linecolor'])
        xtxt = float(i1+(i2-i1)/2.0)
        ax.text(xtxt,ytxt,txt,fontsize=xargs['fontsize'],horizontalalignment='center')  
        
    ax.tick_params(axis='y',labelleft=False,left=False)
    ax.set_ylim([0.,1.])
    ax.set_xlim(xargs['xlim'])    
    
    
def make_row_plot(traces,figsize=(10,4),
            styles=['line'],heights=[1.],xlabels=[''],xscale=[1.],yscale=[1.]):
    """
    Top level function to make a row grid plot
    """
    
    nrows=len(traces)
    if len(styles) < nrows : styles=[styles[0]]*nrows
    if len(heights) < nrows: heights=[heights[0]]*nrows
    if len(xlabels) < nrows: xlabels=[xlabels[0]]*nrows
    if len(xscale) < nrows: xscale=[xscale[0]]*nrows
    if len(yscale) < nrows: yscale=[yscale[0]]*nrows
        
    fig,ax = make_row_grid(heights=heights,figsize=figsize)        
    for i in range(0,nrows): 
        if styles[i] == 'line': add_line_plot(ax[i],traces[i],xscale=xscale[i],yscale=yscale[i])
        elif styles[i] == 'img': add_img_plot(ax[i],traces[i],xscale=xscale[i],yscale=yscale[i])
        elif styles[i] == 'seg': add_seg_plot(ax[i],traces[i],xscale=xscale[i],yscale=yscale[i])
        if (i==0) & (nrows>1):
            ax[i].tick_params(axis='x',labeltop=True,top=True,labelbottom=False,bottom=False) 
        elif i==(nrows-1):
            ax[i].tick_params(axis='x',labelbottom=True)
        else:
            ax[i].tick_params(axis='x',labeltop=False,top=False,labelbottom=False,bottom=False)            
    return(fig,ax)


def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):

  n_channels, n_samples = waveform.shape

  time_axis = np.arange(0, n_samples) / sample_rate

  figure, axes = plt.subplots(n_channels, 1)
  if n_channels == 1:
    axes = [axes]
  for c in range(n_channels):
    axes[c].plot(time_axis, waveform[c], linewidth=1)
    axes[c].grid(True)
    if n_channels > 1:
      axes[c].set_ylabel(f'Channel {c+1}')
    if xlim:
      axes[c].set_xlim(xlim)
    if ylim:
      axes[c].set_ylim(ylim)
  figure.suptitle(title)
  plt.show()
    
    
##########  OBSOLETE FUNCTIONS #################################################################    

# read() is a tiny wrapper around soundfile to read audio data from either local file or url
# - optionally return mono (first channel by default)
# - reading is done to 32bit float's in the range [-1.,1.]
def read(resource,mono=False):

    parsed=urlparse(resource)
    if(parsed.scheme !=''):
        #print(parsed.path)
        fp = io.BytesIO(urlopen(resource).read())
    else:
        #print(parsed.scheme,parsed.path)
        fp = resource
    wavdata, sample_rate = sf.read(fp,dtype='float32')
    if (mono==False):          return(wavdata,sample_rate)
    elif (len(wavdata.shape)==1): return(wavdata,sample_rate)
    else:                      return(wavdata[:,0].flatten(),sample_rate)

    
# a few utilities
# routine for reading audio from different inputs
def read_audio_from_url(url):
  fp = io.BytesIO(urlopen(url).read())
  wavdata, sample_rate = sf.read(fp,dtype='float32')
  # remove extra dim if single channel 
  if wavdata.shape[0] == 1:
    wavdata = wavdata[0]
  return(wavdata,sample_rate)

# by default extract the first channel
def read_mono_from_url(url):
  fp = io.BytesIO(urlopen(url).read())
  wavdata, sample_rate = sf.read(fp,dtype='float32')
  wavdata1 = wavdata[:,1].flatten()
  return(wavdata1,sample_rate)

# time to index converstions;  inputs can be scalars, lists or numpy arrays  outputs are always numpy arrays
def t2indx(t,sample_rate):
  return (np.array(t).astype(float)*float(sample_rate)+0.5).astype(int)
def indx2t(i,sample_rate):
  return np.array(i).astype(float)/float(sample_rate)

#scale=10.0/math.log(10)
DB_EPSILON_KALDI = -69.23689    # scale*math.log(1.19209290e-7)  default flooring applied in KALDI
EPSILON_FLOAT = 1.19209290e-7

