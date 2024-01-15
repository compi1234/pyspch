#  Display Utilities
#
#  Change Log
##############
#  22/09/2021:  plotly backend is broken
#     - the SpchFig constructor cannot easily be made except vai make_subplots
#     - probably this can be fixed in a more recent plotly version with set_subplots
#     - for the time being plotly is not supported
#  14/09/2021: Breaking change in low level API
#     - SpchFig is now a super class of Figure (mpl or plotly)
#     - as such fewer wrapper function are required as the original calls can still be used
#     - a grid of subplots is created as specified, not just rows but also aligned 2D arrangements
#
#  9/6/2021 : Breaking change in arguments of low level API
#     - first argument is 'fig' instead of 'ax'
#     - additional argument 'row' to determine axis selection, counting starts at 1 (=top)
#
#  29/07/2021 & 01/08/2021: many breaking changes
#     - introduction of PYSPCH_BACKEND for choosing the plotting backend  {"mpl","plotly"}
#     - high level API common across backends
#     - choice of low level API on basis of PYSPCH_BACKEND
#     - further adjustment of parameters and options
#         axis numbering and main parameters drawn from matplotlib, plotly is now secondary support
#         low level calls now all start.  my_func(fig,iax,data, ... )
#
#  13/06/2023: prepare PlotSpgFtrs() for 2D layout
#
import os,sys,io 
import scipy.signal

from urllib.request import urlopen

import math
import numpy as np
import pandas as pd

from ..core.constants import EPS_FLOAT, LOG10, SIGEPS_FLOAT

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 

# import low-level API depending on backend
#try: os.environ['PYSPCH_BACKEND'] 
#except: os.environ['PYSPCH_BACKEND'] = "mpl"
#
if os.environ['PYSPCH_BACKEND'] == "mpl":
    from .display_mpl import SpchFig
elif os.environ['PYSPCH_BACKEND']== "plotly":
    print("pyspch(display): using PLOTLY backend !")
    from .display_ly import SpchFig

    
#######################################################################################
# HIGH LEVEL API
#######################################################################################

def PlotWaveform(waveform, sample_rate=8000, title=None, seg=None, ypos=0.8, xlabel="Time(sec)",ylabel=None,xticks=True, yticks=True,color='blue',linewidth=1,**kwargs):
    '''
    Multichannel waveform plotting
    
    '''
    if waveform.size == waveform.shape[0]:
        n_channels = 1
        waveform = waveform.reshape(-1,waveform.size)

    n_channels,n_samples = waveform.shape

    time_axis = np.arange(0, n_samples) / sample_rate

    fig = SpchFig(row_heights=[1.]*n_channels,**kwargs)

    for c in range(n_channels):
        fig.add_line_plot(waveform[c],iax=c,x=time_axis,ylabel=ylabel,xlabel=xlabel,xticks=xticks,yticks=yticks,color=color,linewidth=linewidth)
        if seg is not None:
            fig.add_seg_plot(seg,iax=c,ypos=ypos)
     
    fig.suptitle(title)
    return(fig)

# General Purpose Waveform / Spectrogram Plotting Routine with optional overlayed segmentations
def PlotSpg(spgdata=None,wavdata=None,segwav=None,segspg=None,fig=None,
             sample_rate=1,shift=0.01,y0=0,dy=None,ylabel=None,
             frames=None,ftr_heights=None,title=None,**kwargs):   
    '''General Purpose Waveform / Spectrogram Plotting Routine with optional overlayed segmentations
    
    The screen will consists of 2 axis
        + 1 (TOP):     waveform data (optional)
        + 2:  one or several spectrograms (required)


    If you need more control over the layout, then you need to use the lower level API
    
    Parameters:
    -----------
    wavdata:     waveform data (optional), a 1D numpy array [nsamples]
    spgdata:     spectrogram data (required), numpy array [nparam, nfr] 
    segwav:      segmentation to be added to the waveform plot (optional)
    segspg:      segmentation to be added to the spectrogram plot (optional)
    
    sample_rate: sampling rate (default=1)
    shift:       frame shift for spectrogram (default=0.01)
    y0:          offset on y-axis
    dy:          frequency shift for spectrogram (default=1)
    
    frames:      (int) array [start, end], frame range to show  (optional)
    ylabel:      label for spectrogram frequency axis
    title:       global title for the figure
    
    **kwargs:    optional arguments to pass to the figure creation
        
    '''

    # 1. argument checking and processing


    if(spgdata is None): 
        print("Error(plot): at least a spectrogram needs to be specified")    
    nparam,nfr = spgdata.shape
    if frames is None: frames = [0,nfr]          
    frame_range = np.arange(frames[0],frames[1])
    if dy==None: dy = (sample_rate/2)/(nparam-1)  #assume standard spectrogram
            
    # 2. set up the figure and axis            
    if wavdata is None:
        iax_spg = 0
        heights = [1.]
    else:
        heights = [1.,3.]
        iax_spg = 1
    
    if fig is None:  fig = SpchFig(row_heights=heights,**kwargs)

     
    if sample_rate == 1: # use integer indices
        xlabel = None
    else: # use physical indices
        xlabel = 'Time(secs)'
        
    
    if wavdata is not None:
        n_shift = int(shift*sample_rate)
        # add one extra sample in waveform plot for perfect alignment (if available)
        sample_range = np.arange(frames[0]*n_shift,
                            min(frames[1]*n_shift+1,len(wavdata)) )
        fig.add_line_plot(wavdata[sample_range],iax=0,x=sample_range/sample_rate)
        if segwav is not None:
            fig.add_seg_plot(segwav,iax=0,ypos=0.8,color='#CC0000',size=16)

    fig.add_img_plot(spgdata[:,frame_range],iax=iax_spg,x0=frames[0]*shift+shift/2.,dx=shift,dy=dy,ylabel=ylabel,xlabel=xlabel)
    if segspg is not None:
        fig.add_seg_plot(segspg,iax=iax_spg,ypos=0.9,color='#000000',size=14)

    fig.suptitle(title)

    return fig 

# An Extended Plotting Routine for time aligned

def PlotSpgFtrs(wavdata=None,spgdata=None,segdata=None,line_ftrs=None,img_ftrs=None,row_heights=None,
            spglabel='Frequency (Hz)',line_labels=None, img_labels=None,
            sample_rate=1.,shift=0.01,dy=None,frames=None,Legend=False,**kwargs):
    '''
    General Purpose multi-tier plotting routine of speech signals.
    The figure contains
     - waveform + spectrogram (mandatory)
     - segmentations (list, optional)
     - img features (list, optional)
     - line features (list, optional)
    '''
    if (wavdata is None) or (spgdata is None):
        raise TypeError("PlotSpgFtrs: both wavdata and spgdata are mandatory")
        
    colors=['#000000','#0000CC','#00AA00','#CC0000','#CCAA00','#CC00CC']
    (nparam,nfr)= spgdata.shape
    if frames is None: frames = [0,nfr]
    frame_range = np.arange(frames[0],frames[1])
    frame_times = frame_range*shift + 0.5*shift
    n_shift = int(shift*sample_rate)
    if dy==None: dy = (sample_rate/2)/(nparam-1)  #assume standard spectrogram
    sample_range = np.arange(frames[0]*n_shift,
                        min(frames[1]*n_shift+1,len(wavdata)) )
    # there should be check for singletons here
    #if segdata is not None: segdata = list(segdata)
    #if line_ftrs is not None: line_ftrs = list(line_ftrs)
    #if img_ftrs is not None: img_ftrs = list(img_ftrs)
    nsegs = len(segdata) if segdata is not None else 0
    nlin_ftrs = len(line_ftrs) if line_ftrs is not None else 0  
    nimg_ftrs = len(img_ftrs) if img_ftrs is not None else 0 
    if line_labels== None: line_labels = [None]*nlin_ftrs
    if img_labels== None: img_labels = [None]*nimg_ftrs
    
    if row_heights is None:
        fig = SpchFig(row_heights=[1,3]+nsegs*[.5]+nimg_ftrs*[3.]+nlin_ftrs*[2.],**kwargs)
    else:
        fig = SpchFig(row_heights=row_heights,**kwargs)
    irow = 0
    fig.add_line_plot(wavdata[sample_range],iax=[irow,0],x=sample_range/sample_rate)
    irow = 1
    fig.add_img_plot(spgdata[:,frame_range],iax=[irow,0],x0=(frame_range[0]+0.5)*shift,dx=shift,dy=dy,
                    ylabel=spglabel)
    
    irow = 2
    for i in range(nsegs):
        try:   
            fig.add_seg_plot(segdata[i],iax=[irow,0],ypos=0.5,color=colors[i],Lines=True)
        except:pass
        irow += 1
        
    for i in range(nimg_ftrs):
        try:   
            fig.add_img_plot(img_ftrs[i][:,frame_range],iax=[irow,0],x0=(frames[0]+0.5)*shift,dx=shift,
                           ylabel=img_labels[i] )
        except:pass 
        irow += 1
        
    for i in range(nlin_ftrs):
        try:   
            ftr = line_ftrs[i]
            #print("dim:",ftr.dim)
            if (ftr.ndim == 1): ftr=ftr.reshape(-1,ftr.size)
            #print("shape:",ftr.shape)
            fig.add_line_plot(ftr[:,frame_range],iax=[irow,0],yrange=None,x=frame_times,
                              ylabel=line_labels[i],color=None)
            # a seaborn alternative
            # sns.lineplot(ax=fig.axes[iax],data=line_ftrs[i][:,frame_range].T,legend=Legend);
        except:pass    
        irow += 1
        
    fig.get_axis([irow-1,0]).set_xlabel('Time (sec)') 

    return fig

