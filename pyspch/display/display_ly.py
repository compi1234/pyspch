# !!!!!!!!!! NOT FUNCTIONAL as is !!!!!!!!!!!!!!!!!!
#
#
import os,sys,io 
import scipy.signal
import copy

from urllib.request import urlopen
from IPython.display import display, Audio, HTML, clear_output

import math
import numpy as np
import pandas as pd

import librosa
from pyspch.constants import EPS_FLOAT, LOG10, SIGEPS_FLOAT
import pyspch.spg as specg

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.graph_objects import Figure

os.environ['PYSPCH_BACKEND'] = "plotly"

def iax_2_rc(iax):
    try: iax = list(iax)
    except: iax= [ iax ]
    col = 1 
    if (len(iax) == 1): row = iax[0]+1
    else: row=iax[0]+1; col=iax[1]+1
    return(row,col)



################################
######### LOW LEVEL API
#################################
class SpchFig(Figure):

    def __init__(self,row_heights=[1.,1.],col_widths=[1.],figsize=(12,6),dpi=100,**kwargs):

        super().__init__(**kwargs)

        fig = make_subplots(rows=len(row_heights),cols=len(col_widths),
                row_heights=row_heights,column_widths=col_widths,
                shared_xaxes=True,vertical_spacing=0.04,start_cell='top-left')
        self = copy.deepcopy(fig)
        #self.data = fig.data.copy()
        self.update_layout(width=figsize[0]*dpi, height=figsize[1]*dpi,
                    margin={'b': 0, 'l': 0, 'r': 0, 't': 25},
                    plot_bgcolor="#FCFCFC",paper_bgcolor="White",
                    selectdirection="h")
        
    def add_line_plot(self,y,iax=0,x=None,dx=1.,xrange='tight',yrange='tight',grid='False',color="#0000FF",title=None,xlabel=None,ylabel=None,**kwargs):
        """
        Add a line plot to an existing axis

        Required Parameters
        --------------------
        fig :                 figure object
        iax:                  axis index; either row index or (row,column) index 
                                row index (Numbering: 0=top row)
        y (numpy array):      data as (npts,) or (nftrs,npts) numpy array

        Optional Parameters
        --------------------
        x :        x-axis as (1-D) numpy array (default=None, use sample indices)
        dx :       sampling period on x-axis (default = 1)
                    for time waveforms use dx=1/sample_rate
        xrange :   'tight'(default) or xrange-values 
        yrange :   'tight'(default) or yrange-values. 'tight' on the Y-axis creates 20% headroom
        grid :     False (default)
        xlabel :   default=None
        ylabel :   default=None

        **kwargs : extra kwargs 

        """   

        (row,col) = iax_2_rc(iax)
        if(y.ndim == 1): y=y.reshape(-1,y.size)
        (nftrs,npts)= y.shape

        for j in range(nftrs):
            self.add_trace(go.Scatter(y=y[j,:], x=x, dx=dx,
                          showlegend=False,
                          hoverinfo="x+y",
                          line=dict(color=color)
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
        self.update_xaxes(row=row,title_text=xlabel,range=xrange)
        self.update_yaxes(row=row,title_text=ylabel,fixedrange=True,range=yrange)
        self.update_layout(title_text=title)

    # makes a graphics object (Scatter) of a text transcription
    # segmentation is given as a dataframe with columns [t0,t1,seg]  (t0,t1 expressed in secs)

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

    def add_seg_plot(self,segdf,iax=0,ypos=0.8,color='#EE0000',size=16,Lines=True,linestyle='dash',txtargs={},lineargs={}):

        (row,col) = iax_2_rc(iax)
        textfont={'color':color,'size':size}
        textfont.update(txtargs)

        yax = 'yaxis'+str(row) if (row > 1) else   'yaxis'
        yrange = self['layout'][yax].range
        # if None .. artificial axis range, we assume no ticks desired
        if yrange is None: 
            yrange = (0.,1.)
            self.update_yaxes(row=row,col=col,range=yrange,
                             showticklabels=False,showgrid=False,fixedrange=True)
            self.update_xaxes(row=row,col=col,showgrid=False)
        if ypos is not None:
            _ypos = yrange[0]+ypos*(yrange[1]-yrange[0])
            segdf_go = go_seg_txt(segdf,textfont=textfont,ypos=_ypos)
            self.add_trace(segdf_go,row,col)

        if Lines:
            for iseg in range(0,len(segdf)):
                self.add_vline(row=row,col=col,x=segdf['t0'][iseg],line_dash=linestyle,line_color=color)
                self.add_vline(row=row,col=col,x=segdf['t1'][iseg],line_dash=linestyle,line_color=color)


    def add_img_plot(self,data,iax,dx=1,x0=0,dy=1,y0=0,xlabel=None,ylabel=None): 
        '''
        we are adding xrange and yrange explicitly to the traces
        in doing this we adjust for the fact that we pass center positions to frame data 
        '''    
        (row,col)  = iax_2_rc(iax)
        self.add_trace( go.Heatmap(z=data,dx=dx,x0=x0,dy=dy,
                     colorscale='Jet',
                     showscale=False,
                     showlegend=False,
                     hoverinfo="x+y+z",
                     name='',
                     text=data) , row, col )
        (ny,nx)=data.shape
        #fig.update_xaxes(row=row,title_text=xlabel)
        #fig.update_yaxes(row=row,title_text=ylabel)
        yrange = [0.,(ny-1)*dy]
        self.update_yaxes(row=row,range=yrange,fixedrange=True,title_text=ylabel)
        xrange = [x0 - dx/2.,x0+(nx-1) * dx +dx/2.]
        self.update_xaxes(row=row,range=xrange,title_text=xlabel)


    def add_vlines(self,x,iax=0,color='#F00',linestyle='dashed'):
        '''
        add vertical lines at positions x over full heigth of the axis
        '''
        (row,col) = iax_2_rc(iax)
        mpl2ly={'dashed':'dash','dashdot':'dashdot','dotted':'dot','solid':'solid'}
        for xx in x:
            self.add_vline(row=row,col=col,x=xx,line_dash=mpl2ly[linestyle],line_color=color)

    def add_vrect(self,x0,x1,iax=0,color='#888',alpha=0.2):
        '''
        add vertical rectangle between x0 and x1 
        '''
        self.add_vrect(x0=x0, x1=x1, line_width=0, fillcolor=color, opacity=alpha, row=row,col=col)


    
##### defunct old routines, kept for reference


def update_fig(fig,kv):
    for k,v in kv.items():
        if k == 'title': return
        elif k == 'titlemargin': return

def update_axis(fig,row,kv={}):
    for k,v in kv.items():
        if k == 'xlabel': fig.update_xaxes(row=row+1,title_text=v)
        elif k == 'ylabel': fig.update_yaxes(row=row+1,title_text=v)
        elif k == 'xlim': fig.update_xaxes(row=row+1,range=v)
        elif k == 'ylim': fig.update_yaxes(row=row+1,range=v)
        elif k == 'grid': return

                   

def update_title(fig,title=None,title_margin=50):
    if title is None: return
    fig.update_layout(margin=dict(t=title_margin),title_text=title)
