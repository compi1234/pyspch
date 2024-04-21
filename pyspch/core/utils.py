#  Utilities
#
#
import os,sys,io 

import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 

from .constants import EPS_FLOAT, LOG10, SIGEPS_FLOAT

##################################################
# PART 0:  GLOBAL UTILITIES
##################################################
####
# A single call to check if IN_COLAB has been set
####
def check_colab():
    return('google.colab' in str(get_ipython()))

##################################################
# PART I:  MATH UTILITIES
##################################################

def seconds_to_samples(seconds,sample_rate=8000):
    '''
    convert time in seconds to closest int sample index
    '''    
    return round(float(sample_rate)*seconds)

def round_to_samples(seconds,sample_rate=8000):
    '''
    floating point rounding of time in seconds synchronized to sampling rate
    '''
    samples = round(float(sample_rate)*seconds)
    return float(samples)/float(sample_rate)

def next_power_of_2(x):  
    return 1 if x == 0 else 2**(x - 1).bit_length()

def next_power_of_10(x):  
    return(10. ** math.ceil(math.log10(x)))

def normalize(x, axis=0):
    """Normalizes a multidimensional input array so that the values sums to 1 along the specified axis
    Typically applied to some multinomal distribution

    x       numpy array
            of not normalized data
    axis    int
            dimension along which the normalization should be done

    """

    xs = x.sum(axis,keepdims=True)
    xs[xs<EPS_FLOAT]=1.
    shape = list(x.shape)
    shape[axis] = 1
    xs.shape = shape
    return(x / xs)

def floor(x,FLOOR=EPS_FLOAT):
    """ array floor:  returns  max(x,FLOOR)  """
    return(np.maximum(x,FLOOR))

def logf(x,eps=EPS_FLOAT,logeps=None):
    """ array log with flooring either in log or float"""
    if logeps is None:
        return(np.log(np.maximum(x,eps)))
    else:
        return(np.maximum(np.log(x),logeps))
    
def log10f(x,eps=EPS_FLOAT):
    """ array log10 with flooring """
    return(np.log10(np.maximum(x,eps)))
    
def convertf(x,iscale="lin",oscale="log",eps=EPS_FLOAT):
    """ array conversions between lin, log and log10 with flooring protection """
    if iscale == oscale: 
        return x
    
    if iscale == "lin":
        if oscale == "log":
            return logf(x,eps)
        elif oscale == "log10":
            return log10f(x,eps)
    elif iscale == "log":
        if oscale == "lin":
            return np.exp(x)
        elif oscale == "log10":
            return x/LOG10
    elif iscale == "log10":
        if oscale == "lin":
            return np.power(10.0,x)
        elif oscale == "log":
            return x*LOG10

def dct_diff(dct_a, dct_b):
    dct_a = {k: v for k, v in dct_a.items() if v is not None}
    dct_b = {k: v for k, v in dct_b.items() if v is not None}
    return { k : dct_b[k] for k in set(dct_b.keys()) - set(dct_a.keys()) }
        
##################################################
# PART II: Labels and Segmentations
##################################################
def lbls2seg(lbls,shift=0.01):
    '''
    converts a label sequence (alignment) to a segmentation DataFrame
    This conversion may fail if some frames are not labeled
    
    Arguments:
    lbls       np.array of labels
    shift      frame shift (default=0.01)
    
    Output:
    segmentation dataframe with columns {'t0','t1','seg'}
    '''

    t1 = []
    t0 = [ 0. ]
    seg = [ lbls[0] ]
    for i in range(1,len(lbls)):
        if lbls[i] != lbls[i-1]:
            t0.append( i*shift )
            seg.append(lbls[i])
            t1.append( i*shift )
    t1.append( len(lbls) * shift )
    return( pd.DataFrame({'t0':t0,'t1':t1,'seg':seg}) )   

def seg2lbls(seg, shift=0.01, n_frames=None, end_time=None, pad_lbl=None):
    '''
    seg2lbls() converts a segmentation DataFrame to an alignment 
                in the form of an array of labels
                This conversion will fill empty regions with the padding label
                'pad_lbl' and will make an output at least as large as n_frames (or end_time)
    
    Arguments:
    seg      Segmentation DataFrame with columns {'t0','t1','seg'}
    shift    frame shift (in units used in seg)
    n_frames desired number of frames, overrides last value in seg if larger
    end_time desired end time of segmentation, overrides last value in seg if larger
    pad_lbl  a padding label for missing segments (default=None)
    
    Returns:
    lbls[]   list with labels
    
    '''

    # determine desired end_time/n_frames from arguments or segmentation
    if (n_frames is None) and (end_time is not None):
        n_frames = round(end_time/shift)

    if n_frames is None: 
        n_frames = round(seg['t1'].iloc[-1]/shift)
    # else: n_frames = max(n_frames,n_fr)
        
    lbls = [pad_lbl] * n_frames
    for _, seg in seg.iterrows():
        f0 = max(round(seg['t0']/shift), 0)
        f1 = min(round(seg['t1']/shift), n_frames)
        lbls[f0:f1] = [seg['seg']] * int(f1 - f0)
        
    return lbls 


##################################################
# PART III: Confusion Matrices
##################################################

# Pretty Print routine makes for confusion matrices
def plot_confusion_matrix(cm,labels=[],title='Confusion Matrix\n',figsize=(4,4),norm=False,**kwargs):
    '''
    Plot a Confusion Matrix
    
    Arguments
        cm        confusion matrix, np.array or DataFrame
        labels    Default = []
        norm      boolean (default=False), normalize rows to fractions
        title     Default = 'Confusion Matrix'
        figsize   Default = (4,4)
        **kwargs  extra arguments to pass to sns.heatmap()
    '''
    
    heatmap_args = {
        'annot': True, 
        'fmt': 'd', 
        'annot_kws': {'fontsize': 12, 'color':'k'},
        'xticklabels':labels,
        'yticklabels':labels,
        'square': True, 
        'linecolor': 'k', 
        'linewidth': 1.5, 
        'cmap':'Blues', 
        'cbar': False
        }
    
    heatmap_args.update(kwargs)
    if norm: 
        cm = cm / cm.sum(axis=1)[:,None] 
        heatmap_args['fmt'] = '.2f'
    
    f,ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, **heatmap_args)
    ax.tick_params(axis='y',labelrotation=0.0,left=True)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    return f,ax

def plot_confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none',Diagonal=False, **kwargs):
    """
    adds a covariance confidence ellipse to an existing axis
    
    REF: https://carstenschelp.github.io/2018/09/14/Plot_Confidence_Ellipse_001.html
    
    26/01/2024: modified to optionally force plain diagonal covariances
    
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.
        
    facecolor : str (None)
        facecolor passed to matplotlib.patch.Ellipse

    Diagonal : boolean (False)
        this routine will plot full covariance ellipses, 
        
    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.

    # set pearson correlation to 0.0 if we just use diagonal covariance
    if (Diagonal): pearson = 0.0
    
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = mpl.patches.Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = mpl.transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def plot_confidence_ellipse_0801(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
    """
    adds a covariance confidence ellipse to an existing axis
    
    REF: https://carstenschelp.github.io/2018/09/14/Plot_Confidence_Ellipse_001.html
    
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.
        
    facecolor : str (None)
        facecolor passed to matplotlib.patch.Ellipse
        
    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = mpl.patches.Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = mpl.transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)