# -*- coding: utf-8 -*-
"""
`dtw.py` contain a set of routines for dynamic time warping, including utility functions for plotting, etc.

Created on Feb 13, 2023
        
@author: compi
"""
import numpy as np
import pandas as pd
import math
from scipy.spatial import distance_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import ticker
#
from . import display as Spd
#
trans_LEV = {'dx':[-1,-1,0], 'dy':[-1,0,-1] }
trans_SYM = {'dx':[-1,-1,0], 'dy':[-1,0,-1], 'm':[1.0,1.2,1.2] }
trans_ITA = {'dx':[-1,-1,-1], 'dy':[-1,0,-2], 'm':[1.0,1.2,1.2] }


def dtw(x,y,p=2,trans="SYM",result=None):
    ''' 
    Dynamic Time Warping between Sequences of Vectors
    
        
    Inputs:
    -------
        x           np.array of size (Nx,D) or size (Nx) 
                        Nx being the time axis for sequence x and D the feature vector dimension 
        y           np.array of size (Ny,D) or size (Ny) 
                        Ny being the time axis for sequence y and D the feature vector dimension                         
        p           int (default = 2)
                    power of norm for local distance measure
        trans       str or list (default="SYM")
                    definition of allowed transitions in the dynamic programming search and associated costs

            transitions are specified in a dictionary containing
                'dx': array with delta's of x-backpointers
                'dy': array with delta's of y-backpointers
                'm' : array of multiplicative costs (optional, default=1.0)
                'a' : array of additive costs (optional, default=0.0)

            predefined otions are:
                LEV:      Levenshtein transitions (equivalent to SUB, DEL, INS)
                SYM:      Levenshtein transitions with multiplicative off-diagonal costs
                ITA:      time synchronous Itakura DP   (this is similar to an HMM without skips)
                            allowed moves: (1,0), (1,-1), (1,-2)
                            NOTE: 'Not more than 2 horizontal moves' is not enforced

        result    str, one of "default", "details" or None

    Returns:
    --------
       if result is None or "default":
           dtw_distance     float
           alignment        array of int pairs  [ (ix,iy) ]
           
       if result is "details":
           dtw_distance     float
           alignment        np.array of ints (N,2) (ix.iy) with alignment indices        
           ld_matrix:       np.array of floats (Nx,Ny), local distance matrix
           cd_matrix:       np.array of floats (Nx,Ny), cummulative distance matrix
           bptrs:           np.array of ints (Nx,Ny,2)

    '''
    
    if x.ndim == 1 : x=x.reshape(-1,1)
    if y.ndim == 1 : y=y.reshape(-1,1)
    Nx,D = x.shape
    Ny,Dy = y.shape
    if(D != Dy):
        print("ERROR(dtw): x and y sequences must have the same feature dimension, %d ne %d",D,Dy)
        return(-1)

    shapes = (Nx,Ny)
    
    # define transitions (dx,dy) and transition costs
    if trans == 'SYM':
        trans = trans_SYM
    elif trans[0:3] == 'ITA':
        trans = trans_ITA
    elif trans[0:3] == 'LEV':
        trans = trans_LEV         
    

    dx = np.array(trans['dx'])
    dy = np.array(trans['dy'])
    Ntr = len(dx)
    assert(len(dy) == Ntr)
    m_fac = trans['m'] if 'm' in trans.keys() else np.ones(Ntr)
    a_fac = trans['a'] if 'a' in trans.keys() else np.zeros(Ntr)

    # compute local distance matrix and initialize cd_matrix and backpointer arrays
    ld_matrix = distance_matrix(x, y, p=p)
    cd_matrix = math.inf*np.ones(shapes)
    cd_matrix[0,0] = ld_matrix[0,0]
    bptrs = np.zeros((Nx,Ny,2),'int32')

    
    # DTW recursion
    for i in range(Nx):
        ib = i + dx
        for j in range(Ny):
            if (i,j) == (0,0): continue
            best_score = math.inf
            bptr = [0,0]
            jb = j + dy
            for k in range(Ntr):
                ii = ib[k]
                jj = jb[k]
                if (ii >= 0) & ( jj >= 0) :
                    prev = cd_matrix[ii,jj]
                    new =  prev + m_fac[k]*ld_matrix[i,j] + a_fac[k]
                    if new < best_score:
                        best_score = new
                        bptr = [ii,jj]
            cd_matrix[i,j] = best_score
            bptrs[i,j] = bptr
            
    alignment = backtrace(bptrs)
    dtw_distance = cd_matrix[Nx-1,Ny-1]
    
    if result is None : result = "default"
    if result == "default":
        return(dtw_distance,alignment)
    elif result == "details":
        return(dtw_distance,alignment,ld_matrix,cd_matrix,bptrs)
    
def backtrace(bptrs):
    '''
    finds an alignment by backtracking on an array of backpointers
    
    input:
        bptrs:   np.array of size(Nx,Ny,2)
                    each element contains the backpointer in the cell of the trellis
        
    returns:
        trace:   np.array of size(N,2)
                    is the alignment trace in forward direction
    '''
    (Nx,Ny,_) = bptrs.shape
    (ix,iy) = ( Nx-1,Ny-1 )
    trace = [ (ix,iy) ]
    while True:
        (ix,iy) =  bptrs[ix,iy]
        trace.append( (ix,iy) )
        if (ix == 0)  & (iy == 0): break
    # return the reversed trace
    return( np.array(trace[::-1]) )
 
def warp(x,y,trans='SYM',p=2):
    '''
    warp finds the warped versions of x and y along their alignment path
    The 'trans' and 'p' arguments are passed to the main dtw routine
    
    input:
        x,y          np.arrays of size (Nx,D), (Ny,D)
        
    returns
        x_wp, y_wp   np.arrays of size(N,D), (N,D)
                        with min(Nx,Ny) <= N <= max(Nx,Ny)
    '''
    
    _,wp = dtw(x,y,trans=trans,p=p)
    x_wp = x[wp[:,0]]
    y_wp = y[wp[:,1]]
    return(x_wp, y_wp)


######### plotting routines ##############

def min_max(x,y):
    xmax = x.max() if x is not None else -np.inf
    xmin = x.min() if x is not None else  np.inf
    ymax = y.max() if y is not None else -np.inf
    ymin = y.min() if y is not None else  np.inf
    return min(xmin,ymin), max(xmax,ymax)

def pcolor_heatmap(data,ax=None,annot=False,cmap='Blues',vmin=None,vmax=None,edgecolor=None,linewidth=2,x0=0,y0=0,dx=1,dy=1,**kwargs):
    '''
    A heatmap with annotations based on pcolormesh is plotted on a pre-existing axis
    
    data can be either 
        - numeric:   a regular heatmap will be plotted with optional (annot) annotation
        - string:    the data will be used for annotations in a rectangular grid colored according to facecolor
    
    annot   boolean, to plot annotations or not
    
    arguments passed to pcolormesh:
    cmap (default='Blues'), linewidth (default=2), 
    vmin, vmax, edgecolor, **kwargs
    '''
    Ny,Nx = data.shape
    xi = (np.arange(Nx+1)-.5)*dx + x0
    yi = (np.arange(Ny+1)-.5)*dy + y0
 
    Numeric_Data = True   if np.issubdtype(data.dtype, np.number)  else False 
    
    if Numeric_Data:
        ax.pcolormesh(xi,yi,data,cmap=cmap,shading='flat',vmin=vmin,vmax=vmax,
                      edgecolor=edgecolor,linewidth=linewidth,**kwargs)
        if annot:
            for i in range(Nx):
                for j in range(Ny):
                    ax.text(i,j,"{:.2f}".format(data[j][i]),ha='center')
                    
    else:       
        ax.pcolormesh(xi,yi,np.ones((Ny,Nx)),vmin=0.,vmax=1.,cmap=cmap,shading='flat',
                      edgecolor=edgecolor,linewidth=linewidth,**kwargs)
        for i in range(Nx):
            for j in range(Ny):
                ax.text(i,j,data[j][i],ha='center')
                

def plot_trellis(xy_mat=None,xy_line=None,x=None,y=None,xy_annot=False,ftr_annot=False, bptrs=None,
            ftr_args={},xy_args={},bptr_args={},
            width_ratios=None,height_ratios=None,ftr_scale=.2,fig_width=10.,**kwargs):
    '''
    A plotting routine for sequence comparisons, allowing for the plotting
        - trellis heatmap (eg. local or global distance matrix)
        - sequences x and y with same feature dimension, 
                optional in heatmap or text
        - trace overlay on the heatmap (optional)
        
    The aspect ratio of the distance matrix plot is close to 1.0
    The aspect ratio of the feature plots is close to 1.0 for small d and adjusted for large d,
        to account for approximately 20% of the plotting area
        
    The small upper left square is fig.axes[0] and by default not used, but data can be added later on.
    
    The layout is
            _____________________________________
            |         |                          |
            |         |   x(Nx,d)                |
            |         |                          |
            |---------+--------------------------|
            |         |                          |
            | y(Ny,d) |                          | 
            |         |   xy_mat(Ny,Nx)                       
            |         |                          |            
            |         |                          |
            |---------+--------------------------|

    
    
    inputs:
    -------

    x           np.array of features size (Nx,D) or (Nx)  (default=None)
    y           np.array of features size (Nx,D) or (Ny)  (default=None)
    xy_mat      np.array of floats, size (Nx,Ny)          (default=None)
    xy_line     np.array of int's, size(N,2)              (default=None)

    
    ftr_scale   float, fraction defining space reserved for feature plots (default=0.2)
    width_ratios, height_ratios   gridspec values overriding heuristic settings

    xy_annot    boolean for adding annotations to the trellis
                or array of correct size containing the annotations
    ftr_annot   boolean (default=False). If True, then expects to see features of right dimensions in x and y

    xy_args     **kwargs to be passed to the plotting of the xy dp-trellis
    ftr_args    **kwargs to be passed to the plotting of the x and y feature plots
    
    bptrs       np.array of size(Nx,Ny,2) with backpointers
                plots the backpointers if not None
    


    '''

    Ny,Nx = xy_mat.shape
    N = max(Nx,Ny)
        
    # find proper aspect ratios for the plots given matrix and feature dimensions
    if x is None: dx = 0
    else: 
        if x.ndim == 1 : x=x.reshape(-1,1)
        _,dx = x.shape
    if y is None: dy = 0 
    else:
        if y.ndim == 1 : y=y.reshape(-1,1) 
        _,dy = y.shape
     
    d = max(dx,dy)
    ftr_i = np.arange(d)
    
    if ftr_scale is not None:
        if dx != 0 : dx = ftr_scale*N
        if dy != 0 : dy = ftr_scale*N
    if width_ratios is None:
        width_ratios = [ float(dy)/(Nx+dy), float(Nx)/(Nx+dy) ]
    if height_ratios is None:
        height_ratios = [ float(dx)/(Ny+dx), float(Ny)/(Ny+dx) ]
    
    if N>20 | d>5:
        _ftr_args={'cmap':'YlOrBr','alpha':0.75}
        _xy_args={'cmap':'Blues','alpha':0.75}
    else:
        _ftr_args={'cmap':'YlOrBr','alpha':0.75,'edgecolor':'k','linewidth':2}
        _xy_args={'cmap':'Blues','alpha':0.75,'edgecolor':'k','linewidth':1}  
        _bptr_args={'color':'navy','linewidth':1,'head_width':.1,'head_length':.1}
    
    _ftr_args.update(ftr_args)
    _xy_args.update(xy_args)
    _bptr_args.update(bptr_args)
    
    
    # setup the figure and gridspec and direction and visibility of the axes
    fig_aspect = float(Ny+dx)/float(Nx+dy)
    fig = plt.figure(figsize=(fig_width,fig_width*fig_aspect) )
    gs = fig.add_gridspec(nrows=2,ncols=2,
                          width_ratios=width_ratios,
                          height_ratios=height_ratios ,
                          hspace=0., wspace=0. ) 

    ax0 = fig.add_subplot(gs[0,0])
    ax_x = fig.add_subplot(gs[0,1])
    ax_y = fig.add_subplot(gs[1,0])
    ax_xy = fig.add_subplot(gs[1,1])
    ax_xy.invert_yaxis()
    ax_y.invert_yaxis()
    
    ax0.set_visible(False)
    for ax in [ax_xy, ax_x, ax_y]:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    if x is not None:
        ax_x.xaxis.set_visible(True)
        ax_x.xaxis.set_ticks_position('top')
    else:
        ax_xy.xaxis.set_visible(True)
        ax_xy.xaxis.set_ticks_position('top')    
    if y is not None:
        ax_y.yaxis.set_visible(True)
    else:
        ax_xy.yaxis.set_visible(True)
    
    
    # make a non saturated plot
    vmin = np.min(xy_mat)
    vmax = np.max(xy_mat)
    pcolor_heatmap(xy_mat,ax=ax_xy,vmin=vmin,vmax=vmax,annot=xy_annot,**_xy_args)
    if xy_line is not None:
        ax_xy.plot(xy_line[:,0],xy_line[:,1],lw=2,color='r')           


    
    # find out if data is numeric or not
    Numeric_Data = False
    if x is not None:
        if np.issubdtype(x.dtype, np.number): Numeric_Data = True
    if y is not None:
        if np.issubdtype(y.dtype, np.number): Numeric_Data = True  
                      
    # assure that color maps are identical for x and y feature plots 
    if Numeric_Data:
        vmin,vmax = min_max(x,y) 
        vmax=vmax 
    else:
        vmin = vmax = None

    if x is not None :
        pcolor_heatmap(x.T,ax=ax_x,vmin=vmin,vmax=vmax,annot=ftr_annot,**_ftr_args)
        if Nx <10:
            ax_x.xaxis.set_major_locator(ticker.MaxNLocator(Nx))

    if y is not None:
        pcolor_heatmap(y,ax=ax_y,vmin=vmin,vmax=vmax,annot=ftr_annot,**_ftr_args) 
        if Ny < 10:
            ax_y.yaxis.set_major_locator(ticker.MaxNLocator(Ny))

    # optionally add backpointers to the figure
    if bptrs is not None:
        for i in range(Nx):
            for j in range(Ny):
                if (i==0) and (j==0): continue
                ii,jj = bptrs[i,j]
                ax_xy.arrow(i,j,0.2*(ii-i),0.2*(jj-j),**_bptr_args)     
    plt.close()
    return fig                


def plot_align(x=None,y=None,xy_trace=None,down_sample=1,figsize=(10,4)):
    '''
    plots linear alignment
    '''
    if x.ndim == 1 : x=x.reshape(-1,1)
    if y.ndim == 1 : y=y.reshape(-1,1)
    xrange = [-.5,max(x.shape[0],y.shape[0])-.5]
    fig = Spd.SpchFig(row_heights=[2. , 1., 2.], figsize=figsize )
    
    fig.add_img_plot(y.T,iax=0)   
    fig.add_img_plot(x.T,iax=2)
    for ax in fig.axes:
        ax.set_xlim(xrange)
        ax.axis('off')

    fig.axes[1].set_ylim([-1.,1.])
    ax = fig.axes[1]
    for i in range(0,len(xy_trace),down_sample) :
        ax.plot(xy_trace[i],[-1,1],linestyle='dashed',color='k')        
        
    return(fig)


def plot_align_wav(x_ftrs,y_ftrs,*,x_wav=None,y_wav=None,x_seg=None,y_seg=None,shift=.01,sr=16000,xy_trace=None,down_sample=1):
    ''' plots linear alignment of feature data
    with optional addition of waveform plots
    '''
    
    if x_wav is None or y_wav is None:
        WavPlot = False
        row_heights = [1.,.5,1.]
        figsize = (12,5)
        ax_x = 0
        ax_tr = 1
        ax_y = 2
    else:
        row_heights = [1., 1., .5, 1., 1.]
        WavPlot = True
        figsize = (12,7)
        ax_xwav = 0
        ax_x = 1
        ax_tr =2
        ax_y = 3
        ax_ywav = 4
        
    xrange = np.array([-.5,max(x_ftrs.shape[0],y_ftrs.shape[0])-.5])*shift
    fig = Spd.SpchFig(row_heights=row_heights, figsize=figsize )
    
    if WavPlot:
        fig.add_line_plot(ywavdata,iax=ax_ywav,dx=1/sr)
        fig.add_line_plot(xwavdata,iax=ax_xwav,dx=1/sr)
        if x_seg is not None:
            fig.add_seg_plot(x_seg,iax=ax_xwav,ypos=0.8)
        if y_seg is not None:
            fig.add_seg_plot(y_seg,iax=ax_ywav,ypos=0.2)            
        
    fig.add_img_plot(y_ftrs.T,iax=ax_y,dx=shift)   
    fig.add_img_plot(x_ftrs.T,iax=ax_x,dx=shift)
    for ax in fig.axes:
        ax.set_xlim(xrange)
        ax.axis('off')

    fig.axes[ax_tr].set_ylim([-1.,1.])
    ax = fig.axes[ax_tr]
    for i in range(0,len(xy_trace),down_sample) :
        ax.plot(xy_trace[i]*shift,[1,-1],linestyle='dashed',color='k')        
        
    return(fig)