import os,sys,io 
import scipy.signal

from urllib.request import urlopen
from IPython.display import display, Audio, HTML
import soundfile as sf
import sounddevice as sd

import math
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 

import librosa


# time to index conversions; 
# for synchronization between different shifts, we place points at
#     ti = (i+offs) * dt    with offs=0.5 by default
# inputs can be scalars, lists or numpy arrays  outputs are always numpy arrays
def t2indx(t,dt=1.,align='center'):
    offs = 0.5 if align=='center' else 0.0
    return np.round((np.array(t).astype(float)/float(dt)-offs)).astype(int)
def indx2t(i,dt=1.,align='center'):
    offs = 0.5 if align=='center' else 0.0
    return (np.array(i).astype(float) + offs )*dt
def time_range(n,dt=1.,align='center'):
    offs = 0.5 if align=='center' else 0.0
    return (np.arange(n,dtype='float32')+offs)*dt 

#scale=10.0/math.log(10)
DB_EPSILON_KALDI = -69.23689    # scale*math.log(1.19209290e-7)  default flooring applied in KALDI
EPSILON_FLOAT = 1.19209290e-7
    
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
        
    if output == 'dB':    return(librosa.power_to_db(spg,amin=EPSILON_FLOAT))
    else:                 return(spg)


    
#######################################################################################
######### TOP LEVEL PLOTTING ROUTINES
#######################################################################################
def plot_waveform(waveform, sample_rate, title=None, **kwargs):
    '''
    Multichannel waveform plotting
    
    '''
    if waveform.size == waveform.shape[0]:
        n_channels = 1
        waveform = waveform.reshape(-1,waveform.size)

    n_channels,n_samples = waveform.shape

    time_axis = np.arange(0, n_samples) / sample_rate

    fig,ax = make_row_grid(height_ratios=[1.]*n_channels)

    for c in range(n_channels):
        add_line_plot(ax[c],waveform[c],x=time_axis,ylabel="Channel"+str(c))
    ax[n_channels-1].set_xlabel("Time (sec)")

    if title is not None:
        fig.suptitle(title)


# frames must be specified as [start,end(+1)]
def plot_spg(spg,wav=None,sample_rate=None,shift=0.01,frames=None,segwav=None,segspg=None,y=None,title=None,ylabel=None,**kwargs):   
    '''Plotting routine for standard spectrogram visualization
    
    The screen will consists of 2 parts
        + TOP:     waveform data (optional)
        + BOTTOM:  spectrogram data (required)
    Segmentations can be overlayed on top of either waveform or spectrogram data

    If you need more control over the layout, then you need to use the lower level API
    
    Parameters:
    -----------
    spg:         spectrogram (list or singleton) data (required), numpy array [n_param, n_fr] 
    wav:         waveform data (optional)
    sample_rate: sampling rate (default=None)
                    if None, x-axis is by index; if given, x-axis is by time
    segwav:      segmentation to be added to the waveform plot (optional)
    segspg:      segmentation to be added to the spectrogram plot (optional)
    frames:       (int) array [start, end], frame range to show  (optional)
    y:           (float) array, values for spectrogram frequency axis
    ylabel:      label for spectrogram frequency axis
    title:       global title for the figure
    
    **kwargs:    optional arguments to pass to the figure creation
        
    '''
    if (frames is not None) and (sample_rate is None):
        print("Error(spg_plot): sample_rate must be specified together with a frames[] specification")
        return
        
    if type(spg) is not list: spg = [spg]
    _,nfr = spg[0].shape
    if frames is None: frames = [0,nfr]          
    _frames = np.arange(frames[0],frames[1])
        
    if sample_rate is None: # use integer indices
        dt = None
        wav_xlabel = None
        xfr = None
        dx_segspg = shift
    else: # use physical indices
        dt = 1./sample_rate
        nshift = int(shift*sample_rate)
        wav_xlabel = 'secs'
        xfr = indx2t(_frames,shift)
        dx_segspg = None
    
    heights = [3.]*len(spg)
    if wav is not None:
        heights = [1] + heights
        fig,ax = make_row_grid(height_ratios=heights,**kwargs)
        if(sample_rate is None): 
            _samples = np.arange(0,len(wav))
            xtime = None
        else:                  
            _samples = np.arange(frames[0]*nshift,frames[1]*nshift)
            xtime = _samples/sample_rate
        add_line_plot(ax[0],wav[_samples],x=xtime,xlabel=wav_xlabel)
        iax_spg = 1
        iax_segspg = 1
    else:
        fig,ax = make_row_grid(height_ratios=heights,**kwargs)
        iax_spg=0
        iax_segspg = 0
    for _spg in spg:
        add_img_plot(ax[iax_spg],_spg[:,_frames],x=xfr,ylabel=ylabel,y=y)
        iax_spg+=1
    if segwav is not None:
        add_seg_plot(ax[0],segwav,ylbl=0.8,
                lineargs={'colors':'k','color':'blue'},
                lblargs={'color':'blue','fontsize':14}) 
    if segspg is not None:
        add_seg_plot(ax[iax_segspg],segspg,dx=dx_segspg,ylbl=0.9,
                lineargs={'linestyles':'dotted','color':'white'},
                lblargs={'color':'white','fontsize':14,'fontweight':'bold','backgroundcolor':'darkblue','rotation':'horizontal','ma':'center'}) 
    if title is not None: fig.suptitle(title,fontsize=16);
    fig.align_ylabels(ax[:])
    return fig, ax        
        
        
       
        
############################################################# 
# Elementary plotting utilities for typical multirow plotting
# 
##############################################################

def make_row_grid(height_ratios=[1.,1.],**kwargs):
    """ Create a figure and axis for a multi-row plot
        
    This routine lets you specify the respective row heights.
    Note that some defaults deviate from the mpl defaults such as figsize and dpi

                        
    Parameters
    ----------
    height_ratios :   height ratios for different subplots (array of floats)
    **kwargs :        kwargs to be passed to plt.figure()
                      defaults:  figsize=(12,6), dpi=200, constrained_layout=True
                      
    
    """
    
    fig_kwargs={'clear':True,'constrained_layout':True,'figsize':(12,6),'dpi':200}
    fig_kwargs.update(kwargs)
    
    fig = plt.figure(**fig_kwargs)
    nrows = len(height_ratios)
    gs = fig.add_gridspec(nrows=nrows,ncols=1,height_ratios=height_ratios)
    ax = []
    for i in range(0,nrows):
        axx = fig.add_subplot(gs[i,0])
        ax.append(axx)
    return(fig,ax)

def add_line_plot(ax,y,x=None,dx=1.,xlim='tight',ylim='tight',xlabel=None,ylabel=None,**kwargs):
    """
    Add a line plot to an existing axis
    
    Parameters
    ----------
    ax :       axis where to plot
    y :        data as (1-D) numpy array
    x :        x-axis as (1-D) numpy array (default=None, use sample indices)
    dx :       sample spacing, default = 1.0 ; use dx=1/sample_rate for actual time on the x-axis
    xlim :     'tight'(default) or xlim-values 
    ylim :     'tight'(default) or xlim-values. 'tight' on the Y-axis creates 20% headroom
    xlabel :   default=None
    ylabel :   default=None
    **kwargs : kwargs to be passed to mpl.plot()
    
    """
    
    if x is None: 
        # x = np.arange(len(y)) * dx
        x = time_range(len(y),dx)
    ax.plot(x,y,**kwargs)
    if xlim is None: pass
    elif xlim == 'tight': 
        ddx = (x[-1]-x[0])/len(x)
        ax.set_xlim([x[0]-ddx/2.,x[-1]]+ddx/2.)
    else: ax.set_xlim(xlim)
        
    if ylim is None: pass
    elif ylim == 'tight':
        wmax = 1.2 * max(abs(y)+EPSILON_FLOAT)
        ax.set_ylim(-wmax,wmax)
    else:
        ax.set_ylim(ylim)
        
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
        

def add_img_plot(ax,img,x=None,y=None,xticks=True,xlabel=None,ylabel=None,**kwargs):
    ''' Add an image plot (spectrogram style)
    
    Parameters
    ----------
    ax :     axis
    img :    image
    x,y:     coordinates for X and Y axis points, default = index

    xticks : (boolean) - label the x-axis ticks
    **kwargs: extra arguments to pass / override defaults in plt.imshow()
    
    '''
    
    (nr,nc)= img.shape

    params={'cmap':'jet','shading':'auto'}
    params.update(kwargs)

    # Use x & y center coordinates with same dimensions and centered positions
    if x is None: x = np.arange(nc)
    if y is None: y=  np.arange(nr)
        
    ax.pcolormesh(x,y,img,**params)
    
    if(xticks): ax.tick_params(axis='x',labelbottom=True)
    else:       ax.tick_params(axis='x',labelrotation=0.0,labelbottom=False,bottom=True)         
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
        

def add_seg_plot(ax,df,xlim=None,ylim=None,dx=None,ylbl=0.5,Lines=True,
                 lblargs={},lineargs={}):
    
    '''adds a segmentation to an axis
    
    This can be an axis without prior info; in this case at least xlim should be given to scale the x-axis correctly
    Alternatively the segmentation can be overlayed on an existing plot.  In this case the x and y lim's can be inherited from the previous plot This can be 

    Parameters
    ----------
    ax:         matplotlib axis
    df:         dataframe
    xlim:       X-axis range, if None keep existing 
    ylim:       Y-axis range, if None keep existing 
    dx:         shift to be applied to convert segmentation times to spectrogram frames 
    ylbl:       relative height to print the segmentation labels (default= 0.5)
                of None, do not write out segmentation labels
    Lines:      boolean, to plot segmentation lines (default=True)
    
    **lblargs:  plot arguments for labeling text, passed to ax.text() 
                    such as color, fontsize, ..
    **lineargs: plot arguments for the label lines, passed to ax.vlines()
                    such as linestyles, color, ..
    
    ''' 

    if xlim is not None: ax.set_xlim(xlim)
    else: xlim = ax.get_xlim()
    if ylim is not None: ax.set_ylim(ylim)
    else: ylim = ax.get_ylim()
    if ylbl is not None: 
        ylbl = ylim[0] + ylbl * (ylim[1]-ylim[0])
    
    _lineargs={'linestyles':'solid','colors':'k'}
    _lineargs.update(lineargs)
    _lblargs={'horizontalalignment':'center','fontsize':12,'color':'k'}
    _lblargs.update(lblargs)

    for iseg in range(0,len(df)):
        t0= df['t0'][iseg]
        t1= df['t1'][iseg]
        lbl = df['seg'][iseg]
        if dx is not None:
            t0 = t0/dx-0.5
            t1 = t1/dx-0.5
        if (t0>=xlim[0]) and (t1 <=xlim[1]) :
            if(Lines):
                ax.vlines([t0,t1],ylim[0],ylim[1],**_lineargs)
            if ylbl is not None:
                xlbl = float(t0+(t1-t0)/2.0)
                ax.text(xlbl,ylbl,lbl,**_lblargs)          
   
    
###############################################################################################
# INTERNAL FUNCTIONS
###############################################################################################
def _torch_plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
  ''' From the examples in torchaudio '''
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
        
    
# spectrogram plotting routine with optionally:
#   -- waveform 
#   -- up to 2 segmentations in segmentation panel at the bottom
#   -- optionally a pseudo aligned word transcription in the wav panel
#
def _old_plot_spg(spg=None,wav=None,seg=None,txt=None,figsize=(12,8),spg_scale=2,samplerate=16000,n_shift=160,tlim=None,ShowPlot=True):
    '''plot_spg(): Spectrogram plotting routine
            screen will be built of 3 parts
            TOP:     waveform data (optional) + optional word transcriptions
            MIDDLE:  spectrogram data (at least one required)
            BOTTOM:  segmentations (optional)
    
    Parameters:
        spg         spectrogram (list or singleton) data (required), numpy array [n_param, n_fr] 
        wav         waveform data (optional)
        seg         segmentation (list, singleton or none) plotted in segmentation window at the bottom
                    should be passed as DataFrame, optional
        txt         full segment transcript to be printed in waveform axis
        figsize     figure size (default = (12,8))
        spg_scale   vertical scale of spectrogram wrt wav or seg  (default=2)
        samplerate  sampling rate (default=16000)
        n_shift     frame shift in samples, or equivalently the width of non-overlapping frames
                      this is used for synchronisation between waveform and spectrogram/segmentations
        tlim        segment to render
        ShowPlot    boolean, default=True
                      shows the plot by default, but displaying it can be suppressed for usage in a UI loop
        
     Output:
        fig         figure handle for the plot     


        Notes on alignment:
          The caller of this routine is responsible for the proper alignment between sample stream and frame stream
          (see spectrogram() routine).  By default the full sample stream is plotted.

          spg(n_param,n_fr)    
                  x-range   0 ... nfr-1
                  x-view  [-0.5 , nfr-0.5 ]    extends with +- 0.5
          wavdata(n_samples)
                  x-range   0 ... wavdata
                  x-view    -n_shift/2   nfr*n-shift - n_shift/2   (all converted to timescale)
        '''

    if spg is None:
        print("plot_spg(): You must at least provide a spectrogram")
        return
    if type(spg) is not list: spg = [ spg ]
    nspg = len(spg)
    (n_param,n_fr) = spg[0].shape
    
    if seg is None:
        nseg = 0
    else:
        if type(seg) is not list: seg = [seg]
        nseg = len(seg)
        SegPlot = True       

    WavPlot = False if wav is None   else True
    TxtPlot = False if txt is None   else True
    nwav = 1        if WavPlot       else 0
    
    # make an axes grid for nwav waveform's, nspg spectrogram's, nseg segmentation's
    base_height = 1.0/(nwav+nseg/2.0+nspg*spg_scale)
    nrows = nwav+nspg+nseg
    heights = [base_height]*nrows
    for i in range(0,nspg): heights[nwav+i] = base_height*spg_scale
    for i in range(0,nseg): heights[nwav+nspg+i] = base_height/2.0
    fig = plt.figure(figsize=figsize,clear=True,constrained_layout=True)
    gs = fig.add_gridspec(nrows=nrows,ncols=1,height_ratios=heights)
    
    # frame-2-time synchronization on basis of n_fr frames in spectrogram and n_shift
    #    by default it extends the view at the edges by  1/2 nshift samples
    indxlimits = np.array([-n_shift/2, n_fr*n_shift-n_shift/2])
    tlimits = indx2t(indxlimits,samplerate)  

    # add waverform plot
    if WavPlot:
        ax = fig.add_subplot(gs[0,0])
        n_samples = len(wav)
        # if n_samples <= ((n_fr-1) * n_shift):
        #    print("plot_spg() WARNING: waveform too short for spectrogram: %d <= (%d-1) x %d" %
        #          (n_samples, n_fr,n_shift))
        wavtime = np.linspace(0.0, indx2t(n_samples,samplerate), n_samples)
        ax.plot(wavtime,wav)
        wmax = 1.2 * max(abs(wav)+EPSILON_FLOAT)
        ax.set_ylim(-wmax,wmax)
        fshift = indx2t(n_shift,samplerate)
        ax.set_xlim(tlimits)

        ax.tick_params(axis='x',labeltop=True,top=True,labelbottom=False,bottom=False)
        if TxtPlot:
            ax.text(tlimits[1]/2.,0.66*wmax,txt,fontsize=16,horizontalalignment='center')  

    # add spectrograms
    for i in range(0,nspg):
        ax = fig.add_subplot(gs[nwav+i,0])
        ax.imshow(spg[i],cmap='jet',aspect='auto',origin='lower')
        ax.tick_params(axis='x',labelrotation=0.0,labelbottom=False,bottom=True)        
        if (i == nspg-1) & (nseg==0):
            ax.tick_params(axis='x',labelbottom=True)

    # add segmentations
    for i in range(0,nseg):
        ax = fig.add_subplot(gs[nwav+nspg+i,0])
        _old_plot_seg(ax,seg[i],xlim=tlimits,ytxt=0.5,linestyle='dashed',fontsize=10)
        if i != nseg-1:
            ax.tick_params(axis='x',labelbottom=False)

       # _old_plot_seg(ax,seg1,ymin=0.5,ymax=1.0,ytxt=0.75,linestyle='dashed',fontsize=10)
        # _old_plot_seg(ax_seg,seg2,ymin=0.,ymax=0.5,ytxt=0.25,linecolor='r')

    if not ShowPlot: plt.close()
    return(fig)




######################
# Deprecated Functions
######################

# routine for plotting the segmentations   
def _old_plot_seg(ax,df,xlim=[0.,1.],ytxt=0.5,linestyle='solid',linecolor='k',fontsize=14,Vlines=True):
    ''' !!DEPRECATED!! plot_seg(): plots a segmentation to an axis

    ax:   axis
    df:   dataframe with segment data

    xlim:       X-axis range (default: [0 1])
    [ ymin, ymax: Y-axis range (default: [0 1]) ]
    ytxt        height at which to write out the segmentation (default= 0.5)
    Vlines      flag for plotting segmentation lines (default=True)
    linestyle   default='solid'
    linecolor   default='k'
    fontsize    default=14
    ''' 

    # First plot a dummy axis to avoid matplotlib going wild
    ax.imshow(np.zeros((1,1)),aspect='auto',cmap='Greys',vmin=0.,vmax=1) 
    for iseg in range(0,len(df)):
        i1= df['t0'][iseg]
        i2= df['t1'][iseg]
        txt = df['seg'][iseg]
        if(Vlines):
            ax.vlines([i1,i2],0.,1.,linestyles=linestyle,colors=linecolor)
        xtxt = float(i1+(i2-i1)/2.0)
        ax.text(xtxt,ytxt,txt,fontsize=fontsize,horizontalalignment='center')  
        
    ax.tick_params(axis='y',labelleft=False,left=False)
    ax.set_ylim([0.,1.])
    ax.set_xlim(xlim)
    
    
def _old_make_row_plot(traces,figsize=(10,4),
            styles=['line'],heights=[1.],xlabels=[''],xscale=[1.],yscale=[1.]):
    """
     !!DEPRECATED!!  Top level function to make a row grid plot
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

# xscale here is a hack, as we really should pass xlim
def _old_add_seg_plot(ax,df,xscale=1.,yscale=1.):
    """
     !!DEPRECATED!! Add a segmental plot
    """
    print("waiting for implementation")
    plot_seg(ax,df,xlim=[0.,xscale])
    
def _old_add_img_plot(ax,img,xscale=1.,yscale=1.,xlabel=True):
    """
     !!DEPRECATED!! Add an image plot to a given axis with typical spectrogram layout
    """
    (nr,nc)= img.shape
    print(nr,nc)
    extent = [-0.5, (float(nc)-.5)/xscale, -.5, (float(nr)-.5)/yscale]
    ax.imshow(img,cmap='jet',aspect='auto',origin='lower',extent=extent)

    if(xlabel):
        ax.tick_params(axis='x',labelbottom=True)
    else:
        ax.tick_params(axis='x',labelrotation=0.0,labelbottom=False,bottom=True)     
