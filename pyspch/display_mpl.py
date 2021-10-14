
import os,sys,io 
import scipy.signal

from urllib.request import urlopen
from IPython.display import display, Audio, HTML, clear_output

import math
import numpy as np
import pandas as pd

import librosa
from .constants import EPS_FLOAT, LOG10, SIGEPS_FLOAT

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
from matplotlib.figure import Figure     

os.environ['PYSPCH_BACKEND'] = "mpl"

#######################################################################################
# Define the SpchFig class as a superclass of matplotlib Figure
#######################################################################################
class SpchFig(Figure):
    def __init__(self,row_heights=[1.,1.],col_widths=[1.],**kwargs):
        fig_kwargs={'constrained_layout':True,'figsize':(12,6),'dpi':72}
        fig_kwargs.update(kwargs)
        super().__init__(**fig_kwargs)

        self.nrows = len(row_heights)
        self.ncols = len(col_widths)
        gs = self.add_gridspec(nrows=self.nrows,ncols=self.ncols,
                              height_ratios=row_heights,width_ratios=col_widths)

        for i in range(0,self.nrows):
            for j in range(0,self.ncols):
                ii = i*self.ncols + j
                self.add_subplot(gs[i,j])
                self.axes[ii].Init = False
                
        # we work 100% object oriented and therefore close the plot immediately
        # the figure will only show on demand
        plt.close()
        
        if self.ncols == 1:
            self.align_ylabels(self.axes)
            
# convert list of axis to axis number
    def get_axis(self,iax):
        if isinstance(iax,list): # row*col spec
            ii = iax[0]*self.axes[0].numCols + iax[1]
            ax = self.axes[ii]
        else: # rows only
            ax = self.axes[iax]
        return(ax)    
    
    
#######################################################################################
# Low level API for mpl backend
#######################################################################################
    def add_line_plot(self,y,iax=0,x=None,x0=0.,dx=1.,xrange='tight',yrange=None,grid='False',title=None,xlabel=None,ylabel=None,xticks=True,yticks=True,**kwargs):
        """
        Add a line plot to an existing axis

        Required Parameters
        --------------------
        iax(int):             row index (Numbering: 0=top row)
        y (numpy array):      data as (npts,) or (nftrs,npts) numpy array

        Optional Parameters
        --------------------
        x :        x-axis as (1-D) numpy array (default=None, use sample indices)
        x0:        x-axis offset
        dx :       sampling period on x-axis (default = 1)    
        xrange :   'tight'(default) or xrange-values 
        yrange :   None(default), 'tight' or yrange-values. 'tight' on the Y-axis creates 20% headroom
        grid :     False (default)
        xlabel :   default=None
        ylabel :   default=None

        """

        ax = self.get_axis(iax)
        ax.Init = True
        if(y.ndim == 1): y=y.reshape(-1,y.size)
        nftrs,npts= y.shape

        if x is None: 
            x = x0+np.arange(npts) * dx

        ax.plot(x,y.T,**kwargs)
        if xrange is None: pass
        elif xrange == 'tight': 
            ddx = (x[-1]-x[0])/len(x)
            ax.set_xlim([x[0],x[-1]])
        else: ax.set_xlim(xrange)

        if yrange is None: pass
        elif yrange == 'tight':
            yy = 1.2 * np.max(np.abs(y)+SIGEPS_FLOAT)
            ax.set_ylim(-yy,yy)
        else:
            ax.set_ylim(yrange)

        ax.grid(grid)
        if title is not None: ax.set_title(title)
        if xlabel is not None: ax.set_xlabel(xlabel)
        if ylabel is not None: ax.set_ylabel(ylabel)
        if(yticks): ax.tick_params(axis='y',labelleft=True, left=True)
        else:       ax.tick_params(axis='y',labelleft=False, left=False)
        if(xticks): ax.tick_params(axis='x',labelbottom=True)
        else:       ax.tick_params(axis='x',labelrotation=0.0,labelbottom=False,bottom=True)    


    def add_img_plot(self,img,iax=0,x0=None,y0=None,dx=1,dy=1,x=None,y=None,xticks=True,xlabel=None,ylabel=None,**kwargs):
        ''' Add an image plot (spectrogram style)

        Parameters
        ----------
        iax :    axis number (default=0)
        img :    image, a (nrows x ncols) numpy array
        x,y:     coordinates for X and Y axis points, if None dx,dy are used
        x0, y0:  starting values on x and y axis; if None use dx/2 and dy/2 to center
        dx, dy : int/float (default = 1) 

        xticks : (boolean) - label the x-axis ticks
        xlabel : string (default=None)
        ylabel : string (default=None)
        row :    int (default=1)  [Numbering: row=1=top row]

        **kwargs: extra arguments to pass / override defaults in ax.colormesh()

        '''

        ax = self.get_axis(iax)
        ax.Init = True
        (nr,nc)= img.shape

        params={'cmap':'jet','shading':'auto'}
        params.update(kwargs)

        # Use x & y center coordinates with same dimensions and centered positions
        if x is None: 
            if x0 is None : x0 = 0.5*dx
            x = np.arange(nc) * dx +  x0
        if y is None: 
            if y0 is None : y0 = 0.5*dy
            y=  np.arange(nr) * dy  + y0

        ax.pcolormesh(x,y,img,**params)

        if(xticks): ax.tick_params(axis='x',labelbottom=True)
        else:       ax.tick_params(axis='x',labelrotation=0.0,labelbottom=False,bottom=True)         
        if xlabel is not None: ax.set_xlabel(xlabel)
        if ylabel is not None: ax.set_ylabel(ylabel)

    def add_waterfall2D(self,X,iax=0,ax_ref=0,x0=None,dx=.01,scale=None,colors=['r','g','b','k','y','m']):
        '''
        adds basic 2D waterfall plot in axis iax
        '''
        ax = self.get_axis(iax)
        # check if new axis and do some initialization if needed
        #if ax.Init == False: 
        #    ax.set_xlim(self.axes[ax_ref].get_xlim())
        #    ax.set_ylim([0.,1.])
        #    ax.Init = True
        #    Axis_is_new = True
        #else: 
        #    Axis_is_new = False
            
        xmin = np.min(X)
        xmax = np.max(X)
        if scale is None: scale = -dx/(xmax-xmin)
        if x0 is None: x0 = dx/2

        (nparam,nfr)=X.shape
        ax.set_xlim([x0,x0+nfr*dx])
        for i in range(0,nfr):
            ax.plot(scale*(X[:,i]-xmin)+x0+i*dx,np.arange(nparam),color=colors[i%6])

    def add_seg_plot(self,seg,iax=0,xrange=None,yrange=None,ypos=0.5,Lines=True,Labels=False,color='#FF0000',size=16,ax_ref=0,txtargs={},lineargs={}):

        '''adds a segmentation to an axis

        This can be an axis without prior info; in this case at least xrange should be given to scale the x-axis correctly
        Alternatively the segmentation can be overlayed on an existing plot.  In this case the x and y lim's can be inherited from the previous plot This can be 

        Required Parameters
        -------------------
        iax :       axis number
        seg:        segmentation DataFrame

        Optional Parameters
        -------------------
        xrange:     X-axis range, if None keep existing 
        yrange:     Y-axis range, if None keep existing 
        ypos:       relative height to print the segmentation labels (default= 0.5)
                    if None, do not write out segmentation labels
        ax_ref:     reference axis for timing information (default=0)
        Lines:      boolean, to plot segmentation lines (default=True)
        Labels:     boolean, the segmentation is a label stream (default=False) [not used]
        color:      text and line color (default=#FF0000 (red))
        size:       text size (default=16)

        **txtargs:  extra arguments for labeling text, passed to ax.text() 
                        such as color, fontsize, ..
        **lineargs: extra arguments for the label lines, passed to ax.vlines()
                        such as linestyles, color, ..

        ''' 

        if seg is None: return
        ax = self.get_axis(iax)

        # check if new axis and do some initialization if needed
        if ax.Init == False: 
            ax.set_xlim(self.axes[ax_ref].get_xlim())
            ax.set_ylim([0.,1.])
            ax.Init = True
            Axis_is_new = True
        else: 
            Axis_is_new = False

        if xrange is not None: ax.set_xlim(xrange)
        else: xrange = ax.get_xlim()
        if yrange is not None: ax.set_ylim(yrange)
        else: yrange = ax.get_ylim()
        if ypos is not None: 
            ypos = yrange[0] + ypos * (yrange[1]-yrange[0])

        _lineargs={'linestyles':'dashed','colors':color}
        _lineargs.update(lineargs)
        _txtargs={'horizontalalignment':'center','verticalalignment':'center',
                 'color':color,'fontsize':size}
        _txtargs.update(txtargs)

        if seg is None:
            # just create the axis for future use
            return  
        elif 'seg' in seg.columns:
            # a segmentation dataframe with begin and end times i.e. with (t0,t1,seg) entries
            for iseg in seg.index:
                t0= seg['t0'][iseg]
                t1= seg['t1'][iseg]
                txt = seg['seg'][iseg]
                mid_seg = (t1+t0)/2.

                if ( (xrange[0] < mid_seg) and (mid_seg < xrange[1]) ) :
                    if (ypos is not None) :
                        xpos = float(t0+(t1-t0)/2.0)
                        ax.text(xpos,ypos,txt,**_txtargs)  

                    if ( (xrange[0] < t0) and Lines ) :
                            ax.vlines([t0],yrange[0],yrange[1],**_lineargs)
                    if ( (t1 < xrange[1]) and Lines ) :
                            ax.vlines([t1],yrange[0],yrange[1],**_lineargs)
        elif 'lbl' in seg.columns:
            # a label DataFrame with (t,lbl) entries
            for iseg in seg.index:
                xpos=seg['t'][iseg]
                if (xpos > xrange[0]) and (xpos < xrange[1]) :
                    ax.text(xpos,ypos,seg['lbl'][iseg],**_txtargs)

        # for a new axis, just provide tick marks at the bottom
        if(Axis_is_new): 
            ax.tick_params(axis='y',labelleft=False,left=False) 
            ax.tick_params(axis='x',labelbottom=False,bottom=True)


    def add_vlines(self,x,iax=0,color='#F00',linestyle='dashed'):
        '''
        add vertical lines at positions x over full heigth of the axis
        '''
        ax = self.get_axis(iax)
        y = ax.get_ylim()
        ax.vlines(x,y[0],y[1],colors=color,linestyles=linestyle)


    def add_vrect(self,x0,x1,iax=0,color='#888',alpha=0.2):
        '''
        add vertical rectangle between x0 and x1 
        '''
        ax = self.get_axis(iax)
        ax.axvspan(x0, x1, color=color,  alpha=alpha )
    
    
################################ OLDDDDDDDDDDDDDDDDDDDDD   ############################    
    
def make_subplots(row_heights=[1.,1.],col_widths=[1.],figsize=(12,6),dpi=72,**kwargs):
    """ Create a figure and axis for a multi-row plot
        
    make_subplots has an analogous interface to the make_subplots() in the plotly package
    This routine lets you specify the respective row heights.
    Note that some defaults deviate from the mpl defaults such as figsize and dpi

                        
    Parameters
    ----------
    row_heights :   height ratios for different subplots (array of floats)
    figsize :       figsize in inch. default=(12,6)
    dpi :           scaling factor. default=72
    **kwargs :        kwargs to be passed to plt.figure()
constrained_layout=True
                      
    Returns
    -------
    fig :           Figure
    """
    
    fig_kwargs={'clear':True,'constrained_layout':True,'figsize':figsize,'dpi':dpi}
    fig_kwargs.update(kwargs)
    
    # we like tight x-axis in all situations
    # plt.rcParams['axes.xmargin'] = 0
    nrows = len(row_heights)
    ncols = len(col_widths)
    fig,_ = plt.subplots(nrows=nrows,ncols=ncols,gridspec_kw={'height_ratios':row_heights,'width_ratios':col_widths},
                        **fig_kwargs)
    plt.close()
    return(fig)
    

    
def make_rows(row_heights=[1.,1.],figsize=(12,6),dpi=72,**kwargs):
    """ Create a figure and axis for a multi-row plot
        
    make_rows has an analogous interface to the make_subplots() in the plotly package
    This routine lets you specify the respective row heights.
    Note that some defaults deviate from the mpl defaults such as figsize and dpi

                        
    Parameters
    ----------
    row_heights :   height ratios for different subplots (array of floats)
    figsize :       figsize in inch. default=(12,6)
    dpi :           scaling factor. default=72
    **kwargs :        kwargs to be passed to plt.figure()
constrained_layout=True
                      
    Returns
    -------
    fig :           Figure
    """
    
    fig_kwargs={'constrained_layout':True,'figsize':figsize,'dpi':dpi}
    fig_kwargs.update(kwargs)
    
    # we like tight x-axis in all situations
    plt.rcParams['axes.xmargin'] = 0

    fig = plt.figure(**fig_kwargs)
    nrows = len(row_heights)
    gs = fig.add_gridspec(nrows=nrows,ncols=1,height_ratios=row_heights)


    for i in range(0,nrows):
        fig.add_subplot(gs[i,0])
        fig.axes[i].Init = False  
    plt.close()
    return(fig)

def update_fig(fig,kv):
    for k,v in kv.items():
        if k == 'title': fig.suptitle(v)
        elif k == 'titlemargin': pass

def update_axis(fig,row,kv={}):
    ax = fig.axes[row]
    for k,v in kv.items():
        if k == 'xlabel': ax.set_xlabel(v)
        elif k == 'ylabel': ax.set_ylabel(v)
        elif k == 'xlim': ax.set_xlim(v)
        elif k == 'ylim': ax.set_ylim(v)
        elif k == 'grid': ax.grid(v)
        #elif k == 'box': ax.axis(v)
            
def close_plot(fig):
    fig.align_ylabels(fig.axes)
    plt.close()
