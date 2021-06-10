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

import librosa
from pyspch.constants import EPS_FLOAT, LOG10, SIGEPS_FLOAT
import pyspch.spectrogram as specg

import plotly.graph_objects as go
from plotly.subplots import make_subplots

###########################
########## HIHG LEVEL API
###########################

def plot_waveform(waveform, sample_rate=16000, title=None, seg=None,ypos=0.8, figsize=(8,3),dpi=100,showgrid=False,xlabel="Time(sec)", **kwargs):
    '''
    Multichannel waveform plotting
    
    '''   
    if waveform.size == waveform.shape[0]:
        n_channels = 1
        waveform = waveform.reshape(-1,waveform.size)

    n_channels,n_samples = waveform.shape
    
    fig = make_subplots(rows=n_channels, cols=1, row_heights=[1.]*n_channels,
                   shared_xaxes=True,
                   vertical_spacing=0.02,
                   start_cell='top-left')
    top_margin = 25 if title is None else 50
    fig.update_layout( 
            width=figsize[0]*dpi, height=figsize[1]*dpi,
            margin=dict(l=0, r=0, t=top_margin, b=0),
            paper_bgcolor="White",
        )
    
    for c in range(n_channels):
        add_line_plot(fig,waveform[c],row=c+1,dx=1./sample_rate, title=title, xlabel=xlabel)
        fig.update_yaxes(fixedrange=True,zeroline=False,showgrid=showgrid)  
        fig.update_xaxes(zeroline=False,showgrid=showgrid) 
        
    if seg is not None:
        add_seg_plot(fig,seg,row=1,ypos=ypos)
        
    return(fig)

#########

def plot_spg(spg,fig=None,wav=None,sample_rate=16000,f_shift=0.01,frames=None,segwav=None,ftr_axis=False,ftr_height=1.,segspg=None,title="Spectrogram",figsize=(10,6),dpi=100):

    
    if type(spg) is not list: spg = [spg]
    (nparam,nfr) = spg[0].shape
    if frames is None: frames = [0,nfr]          
    _frames = np.arange(frames[0],frames[1])
    
    if wav is None: row_wav = 0
    else:           row_wav = 1
    heights = [1.]*row_wav + [3.]*len(spg)
    if ftr_axis : heights = heights + [ftr_height]
    nrows = len(heights)
    row_spg = row_wav+1
        
    if fig is None:
        fig = make_subplots(rows=nrows, cols=1, row_heights=heights,
                   shared_xaxes=True,
                   vertical_spacing=0.02,
                   start_cell='top-left')
        top_margin = 25 if title is None else 50
        fig.update_layout( 
                width=figsize[0]*dpi, height=figsize[1]*dpi,
                margin=dict(l=0, r=0, t=top_margin, b=0),
                paper_bgcolor="White",selectdirection="h",
                title_text=title
            ) 
    else:
        fig.data = []  # clear all the data traces
        
    if frames is None: frames = [0,nfr]
    n_shift = int(sample_rate*f_shift)
    
    if wav is not None: 
        x = np.arange(frames[0]*n_shift,frames[1]*n_shift)/sample_rate
        wavrange = [frames[0]*n_shift,frames[1]*n_shift]
        add_line_plot(fig,wav[wavrange[0]:wavrange[1]],x=x,row=row_wav,dx=1./sample_rate,xlabel=None)
        fig.update_yaxes(row=row_wav,zeroline=False,showgrid=False) 
      
        if segwav is not None:
            add_seg_plot(fig,segwav,row=row_wav,ypos=0.8)
  
    for i in range(len(spg)):
        xlabel = 'Time(sec)' if ( i == (len(spg)-1) ) else None
        add_img_plot(fig,spg[i][:,frames[0]:frames[1]],dx=f_shift,x0=(frames[0]+0.5)*f_shift,row=row_spg+i,xlabel=xlabel)
        # yaxis for normal spectrogram would be: dy=sample_rate/(2.*(nparam-1))
        
    if segspg is not None:
        add_seg_plot(fig,segspg,row=row_spg,ypos=.9,textfont={'color':'black','size':16})
    

    return(fig)


################################
######### LOW LEVEL API
#################################

def add_line_plot(fig,y,x=None,dx=1.,xrange='tight',yrange='tight',grid='False',title=None,xlabel=None,ylabel=None,row=1,**kwargs):
    """
    Add a line plot to an existing axis
    
    Parameters
    ----------
    fig :      target figure handle
    y :        data as (1-D) numpy array
    x :        x-axis as (1-D) numpy array (default=None, use sample indices)
    dx :       sample spacing, default = 1.0 ; use dx=1/sample_rate for actual time on the x-axis
    xrange :   'tight'(default) or xrange-values 
    yrange :   'tight'(default) or yrange-values. 'tight' on the Y-axis creates 20% headroom
    grid :     False (default)
    xlabel :   default=None
    ylabel :   default=None
    row :      default=1  (Numbering: 1=top row)
    **kwargs : kwargs to be passed to mpl.plot()
    
    """    
    
    fig.add_trace(go.Scatter(y=y, x=x, dx=dx,
                      showlegend=False,
                      hoverinfo="x+y",
                     ),row,1 )
    
    if yrange == 'tight':
        yy = 1.2*np.max(np.abs(y))
        yrange = [-yy,yy]
        

    if xrange == 'tight':
        if x is None:
            xrange = [0.,float(len(y)-1) * dx]
        else:
            xrange = [x[0],x[-1]]

    # it is best to add these ranges EXPLICITLY in the figure object
    # as you can not querry this later unless via JavaScript
    fig.update_xaxes(row=row,title_text=xlabel,range=xrange)
    fig.update_yaxes(row=row,title_text=ylabel,fixedrange=True,range=yrange)
    fig.update_layout(title_text=title)
    
# makes a graphics object (Scatter) of a text transcription
# segmentation is given as a dataframe with columns [t0,t1,seg]  (t0,t1 expressed in secs)
# frameshift is a multiplication factor that can be applied to convert frame numbers to 
#   secs, typically then 0.01 
def go_seg_txt(segdf,textfont={'color':'black','size':14},ypos=0.5):
    nseg = len(segdf)
    btime = np.array(segdf.t0)
    etime = np.array(segdf.t1)
    _xpos = 0.5*(btime + etime)
    _ypos = ypos*np.ones(nseg)
    segtxt = go.Scatter(x=_xpos,y=_ypos,text=segdf.seg,mode='text',
                  textfont=textfont,
                  hoverinfo="x+text",
                  showlegend=False)
    return segtxt

# The position of the segmentation is relative height in the current axis
def add_seg_plot(fig,segdf,ypos=0.8,textfont={'color':'red','size':18},Lines=True,row=1):

    yax = 'yaxis'+str(row) if (row > 1) else   'yaxis'
    yrange = fig['layout'][yax].range
    if yrange is None: yrange = (0.,1.)
    _ypos = yrange[0]+ypos*(yrange[1]-yrange[0])
    segdf_go = go_seg_txt(segdf,textfont=textfont,ypos=_ypos)
    fig.add_trace(segdf_go,row,1)
    if Lines:
        for iseg in range(0,len(segdf)):
            fig.add_vline(row=row,x=segdf['t0'][iseg], line_dash='dash',line_color='green')
            fig.add_vline(row=row,x=segdf['t1'][iseg], line_dash='dot',line_color='green')


def add_img_plot(fig,data,dx=1,x0=0,dy=1,row=1,col=1,xlabel=None,ylabel=None):
#    hm_go = go.Heatmap(z=data,dx=f_shift,x0=f_shift/2.,dy=sr/(2*(nparam-1)),    
    fig.add_trace( go.Heatmap(z=data,dx=dx,x0=x0,dy=dy,
                 colorscale='Jet',
                 showscale=False,
                 showlegend=False,
                 hoverinfo="x+y+z",
                 name='',
                 text=data) , row, col )
    (ny,nx)=data.shape
    yrange = [0.,(ny-1)*dy]
    fig.update_yaxes(row=row,range=yrange,fixedrange=True,title_text=ylabel)
    xrange = [x0,x0+(nx-1) * dx]
    fig.update_xaxes(row=row,range=xrange,title_text=xlabel)
              
