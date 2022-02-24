"""
File Utilities for accessing TIMIT style files
including conversion of TIMIT phone sets

26/01/2022:  Some corrections in the mapping definitions

21/02/2022:  Addition of TIMIT-41 symbol set, ie. CMU + SIL + CL
"""

import os,sys
import numpy as np
import pandas as pd


######## TIMIT MAPPINGS   61-> 48 -> 39 ##########################

TIMIT48 = ['aa','ae', 'ah','ao','aw','ax','er','ay','b','ch','d','dh','dx','eh','el',
 'm','en','ng','ey','f','g','hh','ih','ix','iy','jh','k','l','n','ow', 'oy','p','r','s','sh','t','th','uh','uw','v','w','y','z','zh','sil','epi','vcl','cl']

# the very rare 'q' is mapped to 'sil' instead of None
# as a variant one could define timit-47 including epi -> silence 
#  15 symbols deleted - 2 symbols added (vcl,cl)
timit61_48={ 
 'ax-h': 'ax',
 'axr': 'er',
 'em': 'm',
 'eng': 'ng',
 'nx': 'n',    
 'hv': 'hh',
 'bcl': 'vcl', 'dcl': 'vcl',  'gcl': 'vcl',
 'kcl': 'cl',  'pcl': 'cl',  'tcl': 'cl',
 'h#': 'sil', 'pau': 'sil',  # ,  'epi': 'sil',    optional would make CMU 47
  'q': 'sil'
}

# only used for scoring purposes for TIMIT phone experiments, where training is done on 48 classes and scoring on 39
#  9 symbols deleted
#
timit48_39= { 
    'vcl':'sil','cl':'sil','epi':'sil',
    'ix':'ih','ax': 'ah' ,'el': 'l', 'en':'n' ,
    'ao': 'aa','zh': 'sh'
}

# rarely used as such
timit61_39 = { 'ao': 'aa','ax': 'ah','ax-h': 'ah','axr': 'er','bcl': 'sil',
 'dcl': 'sil', 'el': 'l', 'em': 'm','en': 'n', 'eng': 'ng', 'epi': 'sil', 'gcl': 'sil',
 'h#': 'sil', 'hv': 'hh', 'ix': 'ih', 'kcl': 'sil', 'nx': 'n', 'pau': 'sil', 
 'pcl': 'sil','q': 'sil','tcl': 'sil','zh': 'sh'}


######## TIMIT  61-> 41  APPROXIMATE CMU MAPPING ##########################
# TIMIT-41 is similar to CMU alphabet mapping, i.e. there are 39 phonemes + SIL + CL
# symbols in TIMIT 41 / CMU, not in TIMIT39:  ao, zh
# symbol  in TIMIT 41, not in CMU or TIMIT39: cl
# symbol  NOT in TIMIT 41/CM, but in TIMIT39: dx
#
# REMARK: it should be understood that the above alphabet mappings do NOT IMPLY that TIMIT transcriptions / segmenations 
# can be mapped to good CMU transcriptions / segmentations
#   - plosives consist of a 'closure' + 'burst'
#   - syllabic mappings should be to symbol sequences, eg. 'el' -> 'ah'+'l'
# 
TIMIT41 = ['aa','ae', 'ah','ao','aw','er','ay','b','ch','d','dh','eh',
 'm','ng','ey','f','g','hh','ih','iy','jh','k','l','n','ow',
 'oy','p','r','s','sh','t','th','uh','uw','v','w','y','z','zh','sil','cl']

timit48_41 = { 
    'epi':'sil',
    'vcl':'cl',
    'dx':'t',
    'ix':'ih','ax': 'ah' ,
    'el': 'l', 'en':'n' 
}
timit61_41={ 
 'axr': 'er',
 'em': 'm',
 'eng': 'ng',
 'nx': 'n',    
 'hv': 'hh',
 'kcl': 'cl',  'pcl': 'cl',  'tcl': 'cl',
 'h#': 'sil', 'pau': 'sil' ,   'q': 'sil', 
## different from 48 mapping
 'bcl': 'cl', 'dcl': 'cl',  'gcl': 'cl',
 'epi': 'sil',
 'dx':'t',
 'ax-h': 'ah', 'ix':'ih','ax': 'ah' ,
 'el': 'l', 'en':'n' 
}


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
    elif(xlat == 'timit61_48') : xlat_dict = timit61_48
    elif(xlat == 'timit61_41') : xlat_dict = timit61_41
    elif(xlat == 'timit61_39') : xlat_dict = timit61_39
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
    
