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
from . import core as Spch
#
TRANSITIONS = {
'LEV': {'dx':[-1,-1,0], 'dy':[-1,0,-1], 'a':[0,1,1], 'm':[1.,0,0.] },
'DTW': {'dx':[-1,-1,0], 'dy':[-1,0,-1] },
'SYM': {'dx':[-1,-1,0], 'dy':[-1,0,-1], 'm':[1.0,1.2,1.2] },
'ITA': {'dx':[-1,-2,-1], 'dy':[-1,-1,-2], 'm':[1.0,2.5,2.5] },
'ITS': {'dx':[-1,-1,0,-2,-1], 'dy':[-1,0,-1,-1,-2], 'm':[1.0,2.0,2.0,1.2,1.2] },
'ITX': {'dx':[-1,-1,-1], 'dy':[-1,0,-2], 'm':[1.0,1.2,2.5] },
'ITY': {'dy':[-1,-1,-1], 'dx':[-1,0,-2], 'm':[1.0,1.2,2.5] },
'ITX3': {'dx':[-1,-1,-1,-1], 'dy':[-1,0,-2,-3], 'm':[1.0,1.2,2.5,4.0] },
'ITY3': {'dy':[-1,-1,-1,-1], 'dx':[-1,0,-2,-3], 'm':[1.0,1.2,2.5,4.0] },

'L2R': {'dx':[-1,-1], 'dy':[-1,0] } 
}

def dtw(x,y,p="sqeuclidean",ld_func=None,trans="DTW",CLIP=False,result=None):
    ''' 
    Dynamic Time Warping between Sequences of Vectors
    
        
    Inputs:
    -------
        x           np.array of size (Nx,D) or size (Nx) 
                        Nx being the time axis for sequence x and D the feature vector dimension 
        y           np.array of size (Ny,D) or size (Ny) 
                        Ny being the time axis for sequence y and D the feature vector dimension                         
        ld_func     named local distance function
                    if None use the internally defined defaults (cfr. 'p')
        
        p           float or str (default: "sqeuclidean")
                        defines the local distance metric
                        str must be one of "sqeuclidean", "euclidean", "hamming"
                        a float returns the Minkowski p-norm
                        [ p=2 is the same as "euclidean" ]

        trans       str or list (default="DTW")
                    definition of allowed transitions in the dynamic programming search and associated costs

            transitions are specified in a dictionary containing
                'dx': array with delta's of x-backpointers
                'dy': array with delta's of y-backpointers
                'm' : array of multiplicative costs (optional, default=1.0)
                'a' : array of additive costs (optional, default=0.0)

            predefined otions are:
                DTW:      Baseline DTW setup where 
                            + the total cost is sum of local distances
                            + allowed backpointers are (-1,-1), (0,-1) and (-1,0)
                LEV:      Levenshtein transitions (equivalent to SUB, DEL, INS)
                            an 'additive' cost of 1.0 is applied to DEL/INS transitions

                L2R:      HMM like left-to-right transitions 
                            time-synchronous on x and self-loop or transition-to-next for y 
                ITA:      symmetric Itakura like DP with max warping of 2 on either axis   
                            allowed backpointers: (-1,-1), (-2,-1), (-1,-2)
                            off-diagonal costs of 1.2
                ITX       Itakura like DP that is time synchronous with X-axis and Y-warping between 0 and 2
                            allowed backpointers: (-1,-1), (0,-1), (-2,-1)
                ITX3      same as ITX with max Y-warping 3
                ITY, ITY3:  similar as ITX, ITX3
                ITS:      Combination of ITA and LEV in which the INS/DEL moves carry a heavy  
                            multiplicative (2.0) cost                            

        result    str, one of "default", "details" or None
        
        CLIP      bool, clip infinite values to something reasonable 

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
    try: 
        trans = TRANSITIONS[trans]
    except:
        pass    

    dx = np.array(trans['dx'])
    dy = np.array(trans['dy'])
    Ntr = len(dx)
    assert(len(dy) == Ntr)
    m_fac = trans['m'] if 'm' in trans.keys() else np.ones(Ntr)
    a_fac = trans['a'] if 'a' in trans.keys() else np.zeros(Ntr)

    # compute local distance matrix and initialize cd_matrix and backpointer arrays
    if ld_func is not None:
        ld_matrix = ld_func(x,y)
    else:
        if p=="hamming":
            ld_matrix = np.zeros((Nx,Ny),dtype='float32')
            for i in range(Nx):
                for j in range(Ny):
                    if x[i] != y[j] : ld_matrix[i,j] = 1.0
        elif p=="sqeuclidean":
            ld_matrix = np.square(distance_matrix(x, y, p=2))
        elif p=="euclidean":
            ld_matrix = distance_matrix(x, y, p=2)
        else:
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
            bptr = [i,j]
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
    # clip inf values in cd_matrix to something close to 2x max in the non infinite values
    if CLIP:
        clip_val =np.max(cd_matrix[np.isfinite(cd_matrix)])
        clip_val = Spch.next_power_of_10(2.*clip_val)
        cd_matrix = np.minimum(cd_matrix,clip_val)
    
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
        (iix,iiy) =  bptrs[ix,iy]
        if ( (iix,iiy) == (ix,iy) ):
            print("ERROR(backtrace): can't backtrack from backpointers at (%d,%d)" % (ix,iy))
            return(None)
        trace.append( (iix,iiy) )
        if (iix == 0)  & (iiy == 0): break
        ix = iix
        iy = iiy
    # return the reversed trace
    return( np.array(trace[::-1]) )
 
def warp(x,y,trace=None,trans='SYM',p=2):
    '''
    warp finds the warped versions of x and y along their alignment path
    The 'trans' and 'p' arguments are passed to the main dtw routine
    
    input:
        x,y          np.arrays of size (Nx,D), (Ny,D)
        trace        warping trace or None
                         if no trace was given ,
                         then one is computed first on basis of parameters 
        
    returns
        x_wp, y_wp   np.arrays of size(N,D), (N,D)
                        with min(Nx,Ny) <= N <= max(Nx,Ny)
                        N = len(trace)
    '''
    
    if trace is None:
        _,trace = dtw(x,y,trans=trans,p=p)
        
    x_wp = x[trace[:,0]]
    y_wp = y[trace[:,1]]
    return(x_wp, y_wp)

def align(x,y,trace,EPS=None,DF=True):
    '''
    computes alignment as (x,y) pairs from x,y and trace as given by dtw() routine
    
    EPS = None:  strict (x,y) pairs are returned   (EPS is None and DF is False is same as 'warp' function)
    EPS = EPSCHAR:  EPSCHAR's are inserted in INSERTION/DELETION positions
    
    returns
    -------
    list of tuples      if DF is False
    panda's dataframe   if DF is True (default)
    '''
    #xy_aligned = [(x[trace[0][0]], y[trace[0][1]])]
    x_al = []
    y_al = []
    edits = []   
    for i in range(len(trace)) :
        x_al.append(x[trace[i][0]])
        y_al.append(y[trace[i][1]])  
        if x_al[i] == y_al[i]:
            edits.append('H')
        else:
            edits.append('S')
    if EPS is None: 
        if DF:
            return(pd.DataFrame({'x':x_al,'y':y_al}))
        else:
            return(x_al, y_al)
    else:
        for i in range(1,len(trace)) :
            if trace[i][1] == trace[i-1][1]:      # INSERTION
                y_al[i] = EPS
                edits[i] = 'I'
            elif trace[i][0] == trace[i-1][0]:    # DELETION
                x_al[i] = EPS 
                edits[i] = 'D'
        if DF:
            return(pd.DataFrame({'x':x_al,'y':y_al,'E':edits}))
        else:
            return(x_al, y_al, edits)

######### plotting routines ##############

def min_max(x,y):
    xmax = x.max() if x is not None else -np.inf
    xmin = x.min() if x is not None else  np.inf
    ymax = y.max() if y is not None else -np.inf
    ymin = y.min() if y is not None else  np.inf
    return min(xmin,ymin), max(xmax,ymax)

def text_heatmap(data,ax=None,annot=False,cmap='Blues',vmin=None,vmax=None,edgecolor=None,linewidth=2,x0=0,y0=0,dx=1,dy=1,
                   fmt="{:.2f}", text_size=None, text_color=None, fontweight=None, bad_color='r', **kwargs):
    ''' 
    A combined heatmap / text-mesh plotting on a pre-existing axis      
        - If the data is numeric: heatmap + optional annotations  
        - If the data is not numeric: no heatmap + annotations   
    The mesh is by definition rectangular and equi-spaced   
    The underlying matplotlib calling function is pcolormesh   
    
    input parameters
    ================
    data       np.array of size (Nrows,Ncols)  
                    this is the same as in pcolormesh
               can be either 
                - numeric:   a regular heatmap will be plotted with optional (annot) annotation
                - string:    the data will be used for annotations in a rectangular grid colored according to facecolor
    
    annot   bool (default: False)
                controls plot of annotations
    
    x0,dx     float (defaults: 0.0, 1.0)   
            offset and sample distance for mesh definition on x-axis (columns)
    y0,dy     float (defaults: 0.0, 1.0)   
            offset and sample distance for mesh definition on y-axis (rows)    
            
    fmt       str  (default: {:.2f})
            print format for numeric data
    
    text_size  str  or None (defalt: None)
            size for printing the annotations.  If None an optimum size is guessed depending on mesh density 
    
    arguments passed to pcolormesh:
    ===============================
    cmap (default='Blues'), linewidth (default=2), vmin, vmax, edgecolor, bad_color, **kwargs
    '''
    
    Ny,Nx = data.shape
    N = max(Nx,Ny)
    xi = (np.arange(Nx+1)-.5)*dx + x0
    yi = (np.arange(Ny+1)-.5)*dy + y0
 
    Numeric_Data = True   if np.issubdtype(data.dtype, np.number)  else False 
    if text_size is None:
        if N > 25:      text_size = "xx-small"
        elif N > 15:    text_size = "x-small"
        elif N > 10:    text_size = "small"
        else:           text_size = "medium"
    
    cmap = plt.get_cmap(cmap).copy()
    cmap.set_bad(bad_color)
    if Numeric_Data:
        ax.pcolormesh(xi,yi,data,cmap=cmap,shading='flat',vmin=vmin,vmax=vmax,
                      edgecolor=edgecolor,linewidth=linewidth,**kwargs)
        if annot:
            for i in range(Nx):
                for j in range(Ny):
                    ax.text(i,j,fmt.format(data[j][i]),ha='center',va='center',size=text_size,fontweight=fontweight,color=text_color)
                    
    else:       
        ax.pcolormesh(xi,yi,np.ones((Ny,Nx)),vmin=0.,vmax=1.,cmap=cmap,shading='flat',
                      edgecolor=edgecolor,linewidth=linewidth,**kwargs)
        for i in range(Nx):
            for j in range(Ny):
                ax.text(i,j,data[j][i],ha='center',va='center',size=text_size,fontweight=fontweight,color=text_color)
                

def plot_trellis(fig=None,xy_mat=None,trace=None,x=None,y=None,xy_annot=False,ftr_annot=False, bptrs=None,
            ftr_args={},xy_args={},bptr_args={},trace_args={},
            width_ratios=None,height_ratios=None,ftr_scale=.2,fig_width=10.,fig_aspect=None,**kwargs):
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
            |         |   x(Nx,d)    --->        |
            |         |                          |
            |---------+--------------------------|
            |         |                          |
            | y(Ny,d) |                          | 
            |    |    |   xy_mat(Nx,Ny).T        |                      
            |    |    |                          |            
            |    +    |                          |
            |---------+--------------------------|

    
    
    inputs:
    -------

    x           np.array of features size (Nx,D) or (Nx)  (default=None)
    y           np.array of features size (Nx,D) or (Ny)  (default=None)
    xy_mat      np.array of floats, size (Nx,Ny)          (default=None)
    trace    np.array of int's, size(N,2)              (default=None)

    
    ftr_scale   float, fraction defining space reserved for feature plots (default=0.2)
    fig_width   float, figure width (default=10.)
    
    width_ratios, height_ratios, fig_aspect   
                these all specify the figure in more detail and are typically computed heuristic settings
                if specified they will override the heuristic values 

    xy_annot    boolean for adding annotations to the trellis
                or array of correct size containing the annotations
    ftr_annot   boolean (default: False). If True, then expects to see features of right dimensions in x and y

    xy_args     **kwargs to be passed to the plotting of the xy dp-trellis
    ftr_args    **kwargs to be passed to the plotting of the x and y feature plots
    bptr_args   **kwargs to be passed to the plotting of the backpointers
    trace_args  **kwargs to be passed to the plotting of the backtrace
    
    bptrs       np.array of size(Nx,Ny,2) with backpointers
                plots the backpointers if not None
    


    '''
    # we assume all data to come in in (x,y) coordinates
    # for text_heatmap based on pcolormesh  we need to arrange the data (y,x) to have x as horizontal and y as vertical axis
    #
    xy_mat = xy_mat.T
    if not isinstance(xy_annot,bool): xy_annot = xy_annot.T
    if not isinstance(ftr_annot,bool): ftr_annot = ftr_annot.T
    
    Ny,Nx = xy_mat.shape
    N = max(Nx,Ny)
        
    # find proper aspect ratios for the plots given matrix and feature dimensions
    if x is None: Dx = 0
    else: 
        if x.ndim == 1 : x=x.reshape(-1,1)
        _,Dx = x.shape
    if y is None: Dy = 0 
    else:
        if y.ndim == 1 : y=y.reshape(-1,1) 
        _,Dy = y.shape
     
    D = max(Dx,Dy)
    ftr_i = np.arange(D)
    
    if ftr_scale is not None:
        if Dx != 0 : Dx = ftr_scale*N
        if Dy != 0 : Dy = ftr_scale*N
    if width_ratios is None:
        width_ratios = [ float(Dy)/(Nx+Dy), float(Nx)/(Nx+Dy) ]
    if height_ratios is None:
        height_ratios = [ float(Dx)/(Ny+Dx), float(Ny)/(Ny+Dx) ]
    
    _bptr_args={'color':'navy','linewidth':1,'head_width':.1,'head_length':.1,
               'length_includes_head':True}
    _trace_args={'color':'red','linewidth':1,'head_width':.1,'head_length':.1,
                'length_includes_head':True}
    _ftr_args={'cmap':'YlOrBr','alpha':0.6,'edgecolor':'k','linewidth':1}
    _xy_args={'cmap':'Blues','alpha':0.6,'edgecolor':'k','linewidth':1}  

    if (N>20) or (D>5):
        _ftr_args={'cmap':'YlOrBr','alpha':0.6}
        _xy_args={'cmap':'Blues','alpha':0.6}
        
    _ftr_args.update(ftr_args)
    _xy_args.update(xy_args)
    _bptr_args.update(bptr_args)
    _trace_args.update(trace_args)    
    
    # setup the figure and gridspec and direction and visibility of the axes
    if fig_aspect is None:
        fig_aspect = float(Ny+Dx)/float(Nx+Dy)
    if fig is None:
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
    text_heatmap(xy_mat,ax=ax_xy,vmin=vmin,vmax=vmax,annot=xy_annot,**_xy_args,**kwargs)
    if trace is not None:
        if N > 100: dot_size = "xx-small"
        elif N > 50: dot_size = "x-small"
        elif N > 25: dot_size = "small"
        else: dot_size = "medium"
        #ax_xy.plot(trace[:,0],trace[:,1],lw=2,color=trace_color)           
        for i in range(len(trace)):
            # ax_xy.text(trace[i,0],trace[i,1],"o",ha='center',va='center',size=dot_size,color=_trace_args['color'])
            if i > 0:
                dx = -(trace[i,0]-trace[i-1,0])
                dy = -(trace[i,1]-trace[i-1,1])
                ax_xy.arrow(trace[i,0]+.15*dx,trace[i,1]+.15*dy,.7*dx,.7*dy,**_trace_args)  
    
    # find out if data is numeric or not
    Numeric_Data = False
    if x is not None:
        if np.issubdtype(x.dtype, np.number): Numeric_Data = True
    if y is not None:
        if np.issubdtype(y.dtype, np.number): Numeric_Data = True  
                      
    # assure that color maps are identical for x and y feature plots 
    if Numeric_Data:
        vmin,vmax = min_max(x,y) 
    else:
        vmin = vmax = None
          
    if x is not None :
        text_heatmap(x.T,ax=ax_x,vmin=vmin,vmax=vmax,annot=ftr_annot,**_ftr_args,**kwargs)
        if Nx <10:
            ax_x.xaxis.set_major_locator(ticker.MaxNLocator(Nx))

    if y is not None:
        text_heatmap(y,ax=ax_y,vmin=vmin,vmax=vmax,annot=ftr_annot,**_ftr_args,**kwargs) 
        if Ny < 10:
            ax_y.yaxis.set_major_locator(ticker.MaxNLocator(Ny))

    # optionally add backpointers to the figure
    if bptrs is not None:
        for i in range(Nx):
            for j in range(Ny):
                if (i==0) and (j==0): continue
                ii,jj = bptrs[i,j]
                # Warning: the fixed offset j-0.2 only works well if unidistance spacing of y is true (default)
                ax_xy.arrow(i,j-0.15,0.25*(ii-i),0.25*(jj-j),**_bptr_args)     
    plt.close()
    return fig                


def plot_trellis2(x,y,left,right,*,bptrs=None,trace=None,figsize=(10,5),**plt_args):
    '''
    A small convenience wrapper around plot_trellis() to plot 2 distance matrices side by side
    The right plot contains bptrs and trace if specified
    
    not extensively tested !!
    '''
    fig = plt.figure( figsize=figsize )
    subfigs = fig.subfigures(1, 2, wspace=.01)
    plot_trellis(fig=subfigs[0],x=x,y=y,xy_mat=left,**plt_args)
    plot_trellis(fig=subfigs[1],x=x,y=y,xy_mat=right,bptrs=bptrs,trace=trace,**plt_args)    
    return(fig)



def plot_align_1(x=None,y=None,trace=None,down_sample=1,figsize=(10,4),cmap='YlOrBr',alpha=0.7):
    '''
    plots linear alignment
    '''
    if x.ndim == 1 : x=x.reshape(-1,1)
    if y.ndim == 1 : y=y.reshape(-1,1)
    xrange = [-.5,max(x.shape[0],y.shape[0])-.5]
    fig = Spd.SpchFig(row_heights=[2. , 1., 2.], figsize=figsize )
    
    fig.add_img_plot(y.T,iax=0,cmap=cmap,alpha=alpha)   
    fig.add_img_plot(x.T,iax=2,cmap=cmap,alpha=alpha)
    for ax in fig.axes:
        ax.set_xlim(xrange)
        ax.axis('off')

    fig.axes[1].set_ylim([-1.,1.])
    ax = fig.axes[1]
    for i in range(0,len(trace),down_sample) :
        ax.plot(trace[i],[-1,1],linestyle='dashed',color='k')        
        
    return(fig)


def plot_align(x,y,trace,*,x_wav=None,y_wav=None,x_seg=None,y_seg=None,shift=.01,sr=16000,
                   down_sample=1,figsize=(10,4),cmap='YlOrBr',alpha=1.0,segcolor='r'):
    ''' 
    plots linear alignment of feature data
    with optional addition of waveform plots
    '''
    if x.ndim == 1 : x=x.reshape(-1,1)
    if y.ndim == 1 : y=y.reshape(-1,1)    
    if x_wav is None or y_wav is None:
        WavPlot = False
        row_heights = [1.,.5,1.]
        #figsize = (12,5)
        ax_x = 0
        ax_tr = 1
        ax_y = 2
    else:
        row_heights = [1., 1., .5, 1., 1.]
        WavPlot = True
        #figsize = (12,7)
        ax_xwav = 0
        ax_x = 1
        ax_tr =2
        ax_y = 3
        ax_ywav = 4
        
    xrange = np.array([-.5,max(x.shape[0],y.shape[0])-.5])*shift
    fig = Spd.SpchFig(row_heights=row_heights, figsize=figsize )
    
    if WavPlot:
        fig.add_line_plot(y_wav,iax=ax_ywav,dx=1/sr)
        fig.add_line_plot(x_wav,iax=ax_xwav,dx=1/sr)
          
    
    #text_heatmap(x.T,ax=fig.get_axis(ax_x),annot=True,cmap=cmap,alpha=alpha)
    fig.add_img_plot(y.T,iax=ax_y,dx=shift,cmap=cmap,alpha=alpha)   
    fig.add_img_plot(x.T,iax=ax_x,dx=shift,cmap=cmap,alpha=alpha)
    if x_seg is not None:
        fig.add_seg_plot(x_seg,iax=ax_x,ypos=0.8,color=segcolor)
    if y_seg is not None:
        fig.add_seg_plot(y_seg,iax=ax_y,ypos=0.2,color=segcolor)  
    for ax in fig.axes:
        ax.set_xlim(xrange)
        ax.axis('off')

    if(trace is not None):
        fig.axes[ax_tr].set_ylim([-1.,1.])
        ax = fig.axes[ax_tr]
        for i in range(0,len(trace),down_sample) :
            ax.plot(trace[i]*shift,[1,-1],linestyle='dashed',color='k')        
        
    return(fig)