#  Utilities
#
#
import os,sys,io 
import scipy.signal

from urllib.request import urlopen
from IPython.display import display, Audio, HTML, clear_output

import math
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 

import librosa
from pyspch.constants import EPS_FLOAT, LOG10, SIGEPS_FLOAT
import pyspch.spectrogram as specg

    
#######################################################################################
# (a) TOP LEVEL PLOTTING ROUTINES
#######################################################################################
def plot_waveform(waveform, sample_rate, title=None, showfig=False,xlabel="Time(sec)",ylabel=None, **kwargs):
    '''
    Multichannel waveform plotting
    
    '''
    if waveform.size == waveform.shape[0]:
        n_channels = 1
        waveform = waveform.reshape(-1,waveform.size)

    n_channels,n_samples = waveform.shape

    time_axis = np.arange(0, n_samples) / sample_rate

    fig = make_subplots(row_heights=[1.]*n_channels,**kwargs)
    ax=fig.axes

    for c in range(n_channels):
        #add_line_plot(ax[c],waveform[c],x=time_axis,ylabel="Channel"+str(c))
        add_line_plot(ax[c],waveform[c],x=time_axis,ylabel=ylabel,xlabel=xlabel)
    ax[n_channels-1].set_xlabel("Time (sec)")

    if title is not None:
        fig.suptitle(title)

    if not showfig: plt.close()
    return(fig)

# frames must be specified as [start,end(+1)]
def plot_spg(spg,fig=None,wav=None,sample_rate=None,f_shift=0.01,frames=None,segwav=None,segspg=None,yax=None,title=None,ylabel=None,showfig=False,**kwargs):   
    '''Plotting routine for standard spectrogram visualization
    
    The screen will consists of 2 parts
        + TOP:     waveform data (optional)
        + BOTTOM:  spectrogram data (required)
    Segmentations can be overlayed on top of either waveform or spectrogram data

    If you need more control over the layout, then you need to use the lower level API
    
    Parameters:
    -----------
    spg:         spectrogram (list or singleton) data (required), numpy array [n_param, n_fr] 
    fig:         figure to plot in (default=None)
    wav:         waveform data (optional)
    sample_rate: sampling rate (default=None)
                    if None, x-axis is by index; if given, x-axis is by time
    segwav:      segmentation to be added to the waveform plot (optional)
    segspg:      segmentation to be added to the spectrogram plot (optional)
    frames:      (int) array [start, end], frame range to show  (optional)
    yax:           (float) array, values for spectrogram frequency axis
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
        dx_segspg = f_shift
    else: # use physical indices
        dt = 1./sample_rate
        n_shift = int(f_shift*sample_rate)
        wav_xlabel = 'Time(secs)'
        xfr = specg.indx2t(_frames,f_shift)
        dx_segspg = None
    
    if(fig is not None): 
        ax = fig.axes
        for axi in ax: axi.cla()
    heights = [3.]*len(spg)
    if wav is not None:
        heights = [1] + heights
        if fig is None:
            fig = make_subplots(row_heights=heights,**kwargs)
            ax = fig.axes
        if(sample_rate is None): 
            _samples = np.arange(0,len(wav))
            xtime = None
        else:                  
            _samples = np.arange(frames[0]*n_shift,frames[1]*n_shift)
            xtime = _samples/sample_rate
        add_line_plot(ax[0],wav[_samples],x=xtime,xlabel=wav_xlabel)
        iax_spg = 1
        iax_segspg = 1
    else:
        if fig is None:
            fig = make_subplots(row_heights=heights,**kwargs)
            ax = fig.axes
        iax_spg=0
        iax_segspg = 0
    for _spg in spg:
        add_img_plot(ax[iax_spg],_spg[:,_frames],x=xfr,ylabel=ylabel,y=yax)
        iax_spg+=1
    if segwav is not None:
        add_seg_plot(ax[0],segwav,ypos=0.8,
                lineargs={'colors':'k','color':'blue'},
                txtargs={'color':'blue','fontsize':14}) 
    if segspg is not None:
        add_seg_plot(ax[iax_segspg],segspg,dx=dx_segspg,ypos=0.9,
                lineargs={'linestyles':'dotted','color':'white'},
                txtargs={'color':'white','fontsize':14,'fontweight':'bold','backgroundcolor':'darkblue','rotation':'horizontal','ma':'center'}) 
    if title is not None: fig.suptitle(title,fontsize=16);
    fig.align_ylabels(ax[:])
    if not showfig: plt.close()
    return fig       
        
        
#######################################################################################
# (b) Low level plotting utilities for multirow plotting
#######################################################################################

def make_subplots(row_heights=[1.,1.],**kwargs):
    """ Create a figure and axis for a multi-row plot
        
    This routine lets you specify the respective row heights.
    Note that some defaults deviate from the mpl defaults such as figsize and dpi

                        
    Parameters
    ----------
    row_heights :   height ratios for different subplots (array of floats)
    **kwargs :        kwargs to be passed to plt.figure()
                      defaults:  figsize=(12,6), dpi=72, constrained_layout=True
                      
    
    """
    
    fig_kwargs={'clear':True,'constrained_layout':True,'figsize':(12,6),'dpi':72}
    fig_kwargs.update(kwargs)
    
    fig = plt.figure(**fig_kwargs)
    nrows = len(row_heights)
    gs = fig.add_gridspec(nrows=nrows,ncols=1,height_ratios=row_heights)
    for i in range(0,nrows):
        fig.add_subplot(gs[i,0])
    return(fig)

def add_line_plot(ax,y,x=None,dx=1.,xrange='tight',yrange='tight',grid='False',title=None,xlabel=None,ylabel=None,**kwargs):
    """
    Add a line plot to an existing axis
    
    Parameters
    ----------
    ax :       axis where to plot
    y :        data as (1-D) numpy array
    x :        x-axis as (1-D) numpy array (default=None, use sample indices)
    dx :       sample spacing, default = 1.0 ; use dx=1/sample_rate for actual time on the x-axis
    xrange :   'tight'(default) or xrange-values 
    yrange :   'tight'(default) or yrange-values. 'tight' on the Y-axis creates 20% headroom
    grid :     False (default)
    xlabel :   default=None
    ylabel :   default=None
    **kwargs : kwargs to be passed to mpl.plot()
    
    """
    
    if x is None: 
        x = np.arange(len(y)) * dx
    ax.plot(x,y,**kwargs)
    if xrange is None: pass
    elif xrange == 'tight': 
        ddx = (x[-1]-x[0])/len(x)
        ax.set_xlim([x[0]-ddx/2.,x[-1]]+ddx/2.)
    else: ax.set_xlim(xrange)
        
    if yrange is None: pass
    elif yrange == 'tight':
        wmax = 1.2 * max(abs(y)+SIGEPS_FLOAT)
        ax.set_ylim(-wmax,wmax)
    else:
        ax.set_ylim(yrange)
        
    ax.grid(grid)
    if title is not None: ax.set_title(title)
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
        

def add_seg_plot(ax,segdf,xrange=None,yrange=None,dx=None,ypos=0.5,Lines=True,
                 txtargs={},lineargs={}):
    
    '''adds a segmentation to an axis
    
    This can be an axis without prior info; in this case at least xrange should be given to scale the x-axis correctly
    Alternatively the segmentation can be overlayed on an existing plot.  In this case the x and y lim's can be inherited from the previous plot This can be 

    Parameters
    ----------
    ax:         matplotlib axis
    segdf:      segmentation DataFrame
    xrange:     X-axis range, if None keep existing 
    yrange:     Y-axis range, if None keep existing 
    dx:         scale to convert spectrogram frame numbers to segmentation units
    ypos:       relative height to print the segmentation labels (default= 0.5)
                of None, do not write out segmentation labels
    Lines:      boolean, to plot segmentation lines (default=True)
    
    **txtargs:  plot arguments for labeling text, passed to ax.text() 
                    such as color, fontsize, ..
    **lineargs: plot arguments for the label lines, passed to ax.vlines()
                    such as linestyles, color, ..
    
    ''' 

    if xrange is not None: ax.set_xlim(xrange)
    else: xrange = ax.get_xlim()
    if yrange is not None: ax.set_ylim(yrange)
    else: yrange = ax.get_ylim()
    if ypos is not None: 
        ypos = yrange[0] + ypos * (yrange[1]-yrange[0])
    
    _lineargs={'linestyles':'solid','colors':'k'}
    _lineargs.update(lineargs)
    _txtargs={'horizontalalignment':'center','fontsize':12,'color':'k'}
    _txtargs.update(txtargs)

    for iseg in range(0,len(segdf)):
        t0= segdf['t0'][iseg]
        t1= segdf['t1'][iseg]
        txt = segdf['seg'][iseg]
        if dx is not None:
            t0 = t0/dx-0.5
            t1 = t1/dx-0.5
        if (t0>=xrange[0]) and (t1 <=xrange[1]) :
            if(Lines):
                ax.vlines([t0,t1],yrange[0],yrange[1],**_lineargs)
            if ypos is not None:
                xpos = float(t0+(t1-t0)/2.0)
                ax.text(xpos,ypos,txt,**_txtargs)          
   
