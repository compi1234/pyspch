"""
File Utilities for accessing TIMIT style files
including conversion of TIMIT phone sets
"""

import os,sys
import numpy as np
import pandas as pd

TIMIT_DEFINTIONS = True
# 61-> cmu is an approximate mapping from TIMIT to CMU dict
# closures are attached to the plosives, further 61->39 mapping except for zh->sh
map61_cmu = { 'bcl':'b','dcl':'d','gcl':'g','pcl':'p','tcl':'t','kcl':'k',
 'epi': 'sil', 'h#': 'sil', 'pau': 'sil', 'q': 'sil',
 'ao': 'aa','ax': 'ah','ax-h': 'ah','axr': 'er',
 'el': 'l', 'em': 'm','en': 'n', 'eng': 'ng', 
  'hv': 'hh', 'ix': 'ih', 'nx': 'n'  
}

map61_39 = { 'ao': 'aa','ax': 'ah','ax-h': 'ah','axr': 'er','bcl': 'sil',
 'dcl': 'sil', 'el': 'l', 'em': 'm','en': 'n', 'eng': 'ng', 'epi': 'sil', 'gcl': 'sil',
 'h#': 'sil', 'hv': 'hh', 'ix': 'ih', 'kcl': 'sil', 'nx': 'n', 'pau': 'sil', 
 'pcl': 'sil','q': 'sil','tcl': 'sil','zh': 'sh'}

map61_48={ 'ax-h': 'ax',
 'axr': 'er',
 'bcl': 'vcl',
 'dcl': 'vcl',
 'em': 'm',
 'eng': 'ng',
 'epi': 'sil',
 'gcl': 'vcl',
 'h#': 'sil',
 'hv': 'hh',
 'kcl': 'cl',
 'nx': 'n',
 'pau': 'sil',
 'pcl': 'cl',
 'q': 'cl',
 'tcl': 'cl',
}
map_closures={'bcl':'b','dcl':'d','gcl':'g','pcl':'p','tcl':'t','kcl':'k'}
map48_39= { 'vcl':'sil','cl':'sil',
'ao': 'aa','ax': 'ah' ,'el': 'l', 'em':'n' ,'zh': 'sh'}

timit_48 = ['aa','ae', 'ah','ao','aw','ax','er','ay','b','vcl','ch','d','dh','dx','eh','el',
 'm','en','ng','epi','ey','f','g','sil','hh','ih','ix','iy','jh','k','cl','l','n','ow',
 'oy','p','r','s','sh','t','th','uh','uw','v','w','y','z','zh']

def read_seg_file(fname,dt=1,fmt=None,xlat=None):
    """
    Routine for reading TIMIT style segmentation files, consisting of lines
        ...
        begin  end  name
        ...    

     begin and end are numbers, but can be expressing different units
     such as samples, time (secs), or frames
     
    Parameters:
    -----------
    fname(str):   file name
    dt(int or float):    sample period to be applied (default=1.)
    fmt (str) :   format for timings (default=None, i.e. inferred from input/dt)
    xlat (str) :  optional phoneme mapping
    
    Returns:
    --------
    segdf(DataFrame):   panda's data frame with columns [t0,t1,seg]
                        if reading of the file fails, None is returned
        
    """
    
    try:
        segdf = pd.read_csv(fname,delim_whitespace=True,names=['t0','t1','seg'])
        segdf['t0'] = segdf['t0']*dt 
        segdf['t1'] = segdf['t1']*dt 
        if fmt is not None:
            segdf['t0'] = segdf['t0'].astype(fmt)
            segdf['t1'] = segdf['t1'].astype(fmt)
        if xlat is not None:
            segdf = xlat_seg(segdf,xlat=xlat)
        return(segdf)
    
    except:
        print(f"WARNING(read_seg_file): reading/converting segmentation file {fname} failed")
        return(None)
    
def write_seg_file(fname,segdf,dt=None,fmt=None):
    """
    write a TIMIT style segmentation to file
    """
    if dt is not None:
        segdf['t0'] = segdf['t0']*dt 
        segdf['t1'] = segdf['t1']*dt 
    if fmt is not None:
        segdf['t0'] = segdf['t0'].astype(fmt)
        segdf['t1'] = segdf['t1'].astype(fmt)         
    nseg = len(segdf)
    fp = open(fname,"w")
    for i in range(nseg):
        fp.write(f"{segdf['t0'][i]} {segdf['t1'][i]}  {segdf['seg'][i]}\n")
    fp.close()

    
def xlat_seg(isegdf,xlat=None):
    """
    convert input segmentation to an output segmentation according
    to a translation dictionary
    consequently merge identical sequential symbols
    
    inputs:
    -------
        isegdf(DataFrame):  input segmentation in panda dataframe format
        xlat(str or dict):  phone translation dictionary (default=None)
        
    outputs:
    --------
        seg(DataFrame):  output segmentation as DataFrame
   
    """

    if(xlat is None):        xlat_dict = None
    elif (xlat == 'map61_48'): xlat_dict = map61_48
    elif (xlat == 'map61_39'): xlat_dict = map61_39     
    elif (xlat == 'map48_39'): xlat_dict = map48_39
    elif (xlat == 'map61_cmu'): xlat_dict = map61_cmu
    else: xlat_dict = xlat
        
    ww=isegdf.seg
    t0=isegdf.t0
    t1=isegdf.t1
    nseg = len(isegdf)
    ww1 = []
    xww = []
    xt0 = []
    xt1 = []
    
    # 1. apply translation dictionary
    if xlat is None:
        ww1 = ww
    else:

        for i in range(0,nseg):
            ww1.append(xlat_dict.get(ww[i],ww[i]))
            
    # 2. merge identical segments
    oseg = 0
    Merge = False
    prev_seg = ""
    for iseg in range(0,nseg):
        if prev_seg == ww1[iseg]:
            xt1[oseg-1]= t1[iseg]            
        else:
            xww.append(ww1[iseg])
            xt0.append(t0[iseg])
            xt1.append(t1[iseg])
            prev_seg = ww1[iseg]
            oseg += 1

    return(pd.DataFrame({'t0':xt0,'t1':xt1,'seg':xww}))
    
