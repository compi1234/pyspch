
import os,sys,io 
import scipy.signal

from urllib.request import urlopen
from IPython.display import display, Audio, HTML, clear_output

import math
import numpy as np
import pandas as pd

import librosa
from ..core.constants import EPS_FLOAT, LOG10, SIGEPS_FLOAT

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
from matplotlib.figure import Figure     
from matplotlib import ticker

###########################
# 31/10/2023 for v0.8  with matplotlib v3.7
# FutureWarnings are suppressed due to annoying warning in matplotlib's  ax.text() 
# This is obviously NOT future proof
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
############################

os.environ['PYSPCH_BACKEND'] = "mpl"


# local utilities
def invert_xy_line2D(ax,swap_labels=True):
        
    n_lines = len(ax.lines)
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    xlabel = ax.get_xlabel()
    ylabel = ax.get_ylabel()
    xdata = []
    ydata = []
    
    for i in range(n_lines):
        xdata.append(ax.lines[i].get_xdata())
        ydata.append(ax.lines[i].get_ydata())
        
    ax.clear()

    for i in range(n_lines):
        ax.plot(ydata[i],xdata[i])

        
    ax.set_ylim(x_lim)
    ax.invert_yaxis()
    ax.set_ylabel(xlabel,rotation=0)
    ax.set_xlim(y_lim)
    #ax.invert_xaxis()
    ax.set_xlabel(ylabel,rotation=0)    
    
    if swap_labels:
        #ax.yaxis.tick_right()
        #ax.yaxis.set_label_position('right') 
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top') 


#######################################################################################
# Define the SpchFig class as a superclass of matplotlib Figure
#######################################################################################
class SpchFig(Figure):
    '''
    The SpchFig class creates an 2D array of plots (nrows x ncols) [default 2x1]
    The axis numbering is from top to bottom, left to right, thus
    
            0,0    0,1    0,2
            1,0    1,1    1,2
            ...

    v0.8.2.: the default dpi has been increased to 100 (from 72) 
    default figsize is change to (12,7) from (12,6)  which is close to 16:9 aspect ratio
    A larger figsize will yield better pictures for reusage, but might require adjusting the fonts for optimal readability
    '''
    
    def __init__(self,row_heights=[1.,1.],col_widths=[1.],sharex=False,sharey=False,**kwargs):
        fig_kwargs={'constrained_layout':True,'figsize':(12,6),'dpi':100}
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
            

    def get_axis(self,iax):
        '''
        converts axis grid specification to axis number
        '''
        if isinstance(iax,list): # row*col spec
            # deprecated in matplotlib 3.4
            # ii = iax[0]*self.axes[0].numCols + iax[1]
            nc = self.axes[0].get_gridspec().ncols
            ii = iax[0]*nc + iax[1]
            ax = self.axes[ii]
        else: # rows only
            ax = self.axes[iax]
        return(ax)    
    
    
#######################################################################################
# Low level API for mpl backend
#######################################################################################
# this function redraws all lines in a standard line2D plot while inverting y_x
#


    def add_line_plot(self,y,iax=0,x=None,x0=0.,dx=1.,xrange='tight',yrange=None,grid='False',title=None,xlabel=None,ylabel=None,xticks=True,yticks=True,invert_xy=False,color=None,linewidth=1,**kwargs):
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
        color :    default=None (mpl color cycling will be used in successive plots)

        """

        ax = self.get_axis(iax)
        ax.Init = True
        if(y.ndim == 1): y=y.reshape(-1,y.size)
        _,npts= y.shape

        if x is None: 
            x = x0+np.arange(npts) * dx

        ax.plot(x,y.T,color=color,linewidth=linewidth,**kwargs)
        if xrange is None:         pass
        elif xrange == 'tight':    ax.set_xlim([x[0],x[-1]])
        else:                      ax.set_xlim(xrange)

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
            
        if invert_xy:
            invert_xy_line2D(ax,swap_labels=True)


    def add_img_plot(self,img,title=None,iax=0,x0=0,y0=0,dx=1,dy=1,x=None,y=None,xticks=True,xlabel=None,ylabel=None,xtick_align='center',ytick_align='center',**kwargs):
        ''' Add an image plot (spectrogram style)

        Parameters
        ----------
        iax :    axis number (default=0)
        img :    image, a (nrows x ncols) numpy array  [or (,ncols)]
        x,y:     coordinates for X and Y axis points, if None inferred from dx,dy
        x0, y0:  starting values on x and y axis; these are the centerpoints of the patches; default = index positions
                    (USED TO BE:  dx/2 and dy/2 )
        dx, dy : int/float (default = 1, i.e. indices) 

        xticks : (boolean) - label the x-axis ticks
        xlabel : string (default=None)
        ylabel : string (default=None)
        
        xtick_align : default = None , 'center'
                other options: 'edge'
        ytick_align : default = None , 'center'
                other options: 'edge'

        **kwargs: extra arguments to pass / override defaults in ax.colormesh()

        '''
        
        if x is not None:
            print("WARNING(add_img_plot): specifying x is OBSOLETE - Resetting !!")
            x = None
        if y is not None:
            y = None
            print("WARNING(add_img_plot): specifying y is OBSOLETE - Resetting !!")

        ax = self.get_axis(iax)
        ax.Init = True
        # resize a 1D data array to a horizontal strip
        if img.ndim == 1: img=img.reshape(-1,1)
        (nr,nc)= img.shape

        params={'cmap':'jet','shading':'flat'}
        params.update(kwargs)

        # code before v0.6
        # Use x & y center coordinates with same dimensions and centered positions
        #if x is None: 
        #    if x0 is None : x0 = 0.5*dx
        #    x = np.arange(nc) * dx +  x0
        #if y is None: 
        #    if y0 is None : y0 = 0.5*dy
        #    y=  np.arange(nr) * dy  + y0

        # give x,y as edges, thus dim+1 vs. img
        #if x0 is None : x0 = .5*dx
        #if y0 is None : y0 = .5*dy
        # convert from center points to edge defintions for pcolormesh
        #x = x0 -.5*dx + dx * np.arange(nc+1)
        #y = y0 -.5*dy + dy * np.arange(nr+1)
        
        # adjusted 24/01/2023
        # assume x and y are center positions
        #if x is None:
        #    if x0 is None: x0 = 0
        #    x = np.arange(nc)*dx + x0
        #if y is None:
        #    if y0 is None: y0 = 0
        #    y = np.arange(nr)*dy + y0   
        
        # give x,y as edges, thus dim+1 vs. img
        # 
        #if x0 is None : x0 = .5*dx
        #if y0 is None : y0 = .5*dy
        # convert from center points to edge defintions for pcolormesh
        # and flat shading option
        x = x0 -.5*dx + dx * np.arange(nc+1)
        y = y0 -.5*dy + dy * np.arange(nr+1)        
        ax.pcolormesh(x,y,img,**params)
        if xtick_align is None: xtick_align = 'center'
        if xtick_align == 'edge':
            ax.xaxis.set_major_locator(ticker.IndexLocator(base=round(nc/8)*dx,offset=0))
        if ytick_align is None: ytick_align = 'center'
        if ytick_align == 'edge':
            ax.yaxis.set_major_locator(ticker.IndexLocator(base=round(nr/8)*dx,offset=0))
            
        if(xticks): ax.tick_params(axis='x',labelbottom=True)
        else:       ax.tick_params(axis='x',labelrotation=0.0,labelbottom=False,bottom=True)         
        if xlabel is not None: ax.set_xlabel(xlabel)
        if ylabel is not None: ax.set_ylabel(ylabel)
        if title is not None: ax.set_title(title)
        
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

    def add_seg_plot(self,seg,iax=0,title=None,xrange=None,yrange=None,ypos=0.5,Lines=True,Labels=False,color='#FF0000',size=16,ax_ref=0,txtargs={},lineargs={}):

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

        if title is not None: ax.set_title(title)

    def add_vlines(self,x,iax=0,color='#F00',linestyle='dashed', **kwargs):
        '''
        add vertical lines at positions x over full heigth of the axis
        '''
        iaxes = np.atleast_1d(iax)
        xx = np.atleast_1d(x)
        for iax in iaxes:
            ax = self.get_axis(iax)
            y = ax.get_ylim()
            for x in xx:
                ax.vlines(x,y[0],y[1],colors=color,linestyles=linestyle, **kwargs)

    def add_vrect(self,x0,x1,iax=0,color='#AAA',ec="#000",lw=2.,alpha=1.,fill=False,**kwargs):
        '''
        add vertical rectangle between x0 and x1 
        '''
        iaxes = np.atleast_1d(iax)
        for iax in iaxes:
            ax = self.get_axis(iax)
            ax.axvspan(x0, x1, color=color, fill=fill, lw=lw, ec=ec, alpha=alpha , **kwargs)
    

    def remove_patches(self,iax=None):
        '''
        remove patches from the figure or specified axes
        '''
        if iax is None:
            for ax in self.axes:
                ax.patches =[]
        else:
            iaxes = np.atleast_1d(iax)
            for iax in iaxes:
                ax = self.get_axis(iax)
                ax.patches =[]        
        
        
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
