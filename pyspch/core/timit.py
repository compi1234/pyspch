"""
File Utilities for accessing TIMIT style files
including conversion of TIMIT phone sets

26/01/2022:  Some corrections in the mapping definitions

21/02/2022:  Addition of TIMIT-41 symbol set, ie. CMU + SIL + CL
06/02/2023:  Allowing equivalence names TIMIT41 or CMU


TIMIT PHONE SETS:   
The different phone sets defined here all use ARPABET notations but differ slightly in the phonetic detail that is maintained.   

TIMIT61: alphabet used in TIMIT transcriptions
    seldomly used in ASR   
    
TIMIT48: alphabet typically used for ASR training on TIMIT database  
    - 7 phonetic symbols are folded into other existing phones
    - 9 "silence"-like symbols are folded into 3 NEW symbols (SIL, CL, VCL)
    
TIMIT39: alphabet typically used for SCORING in ASR experiments with TIMIT
    compared to TIMIT48
        - 3 silence-like labels are folded onto SIL
        - 6 phone like labels are folded onto a similar phone

CMU or TIMIT41: CMU39 + sil + cl
    - phones are identical to CMU39 + 'sil' + 'cl' ('cl' is optional and does not appear in CMU)
    - comparing to TIMIT48
        - closures are mapped to 'cl'  (this is an additional symbol vs. CMU)
        - DX is folded onto T
    - comparing to TIMIT39
        - AO and ZH are preserved
    
REMARK: it should be understood that the above alphabet mappings do NOT IMPLY that TIMIT transcriptions / segmenations 
 can be mapped to good CMU transcriptions / segmentations
   - plosives consist of a 'closure' + 'burst'
   - syllabic mappings should be to symbol sequences, eg. 'el' -> 'ah'+'l'
"""

import os,sys,re
import numpy as np
import pandas as pd
import pkg_resources
from .file_tools import *


#### EXAMPLE WORDS for ARPABET PHONEMES (CMU set)
arpabet2word={"aa":"balm","ae":"bat","ah":"butt","ao":"bought","aw":"bout",
              "ay":"bite","b":"bite","ch":"church","d":"die","dh":"thy",
              "eh":"bet","er":"bird","ey":"bait","f":"fight","g":"guy",
              "hh":"high","ih":"bit","iy":"beat","jh":"jive","k":"kite",
              "l":"lie","m":"my","n":"nigh","ng":"sing","ow":"boat","oy":"boy",
               "p":"pie","r":"rye","s":"sigh","sh":"shy","t":"tie","th":"thigh",
               "uh":"book","uw":"boot","v":"vie","w":"why","y":"yacht","z":"zoo",
                     "zh":"pleasure"}
pb2word={"aa":"hod","ae":"had","ah":"hud","ao":"hawed","ay":"hide",
              "eh":"head","er":"heard","ey":"haid","ih":"hid","iy":"heed",
        "uh":"hood","uw":"who'd"}
                

######## TIMIT MAPPINGS   61-> 48 -> 39 ##########################

TIMIT61 = ['aa','ae', 'ah','ao','aw','ax','ax-h', 'axr', 'ay', 'b', 'bcl',
 'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey', 'f',
 'g', 'gcl', 'h#', 'hh','hv', 'ih','ix', 'iy', 'jh', 'k', 'kcl', 'l', 'm',
 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl', 'q', 'r', 's', 'sh', 't', 'tcl','th', 'uh','uw', 'ux',
 'v', 'w', 'y', 'z', 'zh']

TIMIT48 = ['aa','ae', 'ah','ao','aw','ax','er','ay','b','ch','d','dh','dx','eh','el',
 'm','en','ng','ey','f','g','hh','ih','ix','iy','jh','k','l','n','ow', 'oy','p','r','s','sh','t','th','uh','uw','v','w','y','z','zh','sil','epi','vcl','cl']

TIMIT41 = ['aa','ae', 'ah','ao','aw','er','ay','b','ch','d','dh','eh',
 'm','ng','ey','f','g','hh','ih','iy','jh','k','l','n','ow',
 'oy','p','r','s','sh','t','th','uh','uw','v','w','y','z','zh', 'sil','cl']

TIMIT39 = ['aa','ae', 'ah','aw','er','ay','b','ch','d','dh','dx','eh',
 'm','ng','ey','f','g','hh','ih','iy','jh','k','l','n','ow',
 'oy','p','r','s','sh','t','th','uh','uw','v','w','y','z','sil']

CMU = TIMIT41

# TIMIT 61 -> 48
################
# The TIMIT-47 variant is obtained by also mapping epi -> silence 
timit61_48_diff ={ 
 'ax-h': 'ax',
 'axr': 'er',
 'em': 'm',
 'eng': 'ng',
 'hv': 'hh',
 'nx': 'n',    
 'ux': 'uw',
 'bcl': 'vcl', 'dcl': 'vcl',  'gcl': 'vcl',
 'kcl': 'cl',  'pcl': 'cl',  'tcl': 'cl',
 'h#': 'sil', 'pau': 'sil',  # ,  'epi': 'sil',    optional would make CMU 47
  'q': 'sil'
}

# only used for scoring purposes for TIMIT phone experiments, where training is done on 48 classes and scoring on 39
#  9 symbols deleted;   deleting the 'ao' and 'zh' symbols seems a bit erratic - these are maintained in TIMIT41
#
timit48_39_diff = { 
    'vcl':'sil','cl':'sil','epi':'sil',
    'ix':'ih','ax': 'ah' ,'el': 'l', 'en':'n' ,
    'ao': 'aa','zh': 'sh'
}

# rarely used as such
timit61_39_diff = { 'ao': 'aa','ax': 'ah','ax-h': 'ah','axr': 'er','bcl': 'sil',
 'dcl': 'sil', 'el': 'l', 'em': 'm','en': 'n', 'eng': 'ng', 'epi': 'sil', 'gcl': 'sil',
 'h#': 'sil', 'hv': 'hh', 'ix': 'ih', 'kcl': 'sil', 'nx': 'n', 'pau': 'sil', 
 'pcl': 'sil','q': 'sil','tcl': 'sil','zh': 'sh'}


######## TIMIT  61-> 41  APPROXIMATE CMU MAPPING ##########################
#

# 


timit48_41_diff = { 
    'epi':'sil',
    'vcl':'cl',
    'dx':'t',
    'ix':'ih','ax': 'ah' ,
    'el': 'l', 'en':'n' 
}
timit61_41_diff ={ 
 'axr': 'er',
 'em': 'm',
 'eng': 'ng',
 'nx': 'n',    
 'hv': 'hh',
 'ux': 'uw',
 'kcl': 'cl',  'pcl': 'cl',  'tcl': 'cl',
 'h#': 'sil', 'pau': 'sil' ,   'q': 'sil', 
## different from 48 mapping
 'bcl': 'cl', 'dcl': 'cl',  'gcl': 'cl',
 'epi': 'sil',
 'dx':'t',
 'ax-h': 'ah', 'ix':'ih','ax': 'ah' ,
 'el': 'l', 'en':'n' 
}

# FILE based definitions of the TIMTI alphabet and their mappings
# This usage is the preferred way of doing this

####################################################################
# TIMIT ALPHABETS and MAPPINGS
####################################################################

def get_arpa_word(label):
    '''
    return a word example for an arpabet symbol
    '''
    
    return(arpabet2word[label])

def get_timit_alphabet(labset="timit61"):
    '''
    gets one of the various TIMIT alphabets
    acceptable names: timit61, timit48, timit41, timit39  (upper or lowercase allowed)
    '''
    timit_map = {
        "timit61": TIMIT61,
        "timit48": TIMIT48,
        "timit39": TIMIT39,
        "timit41": TIMIT41,
        "cmu":CMU
        }
    
    return(timit_map[labset])

def get_timit_mapping(mapping=None,set1="timit61",set2="cmu"):
    '''
    makes a mapping dictionary between various TIMIT alphabets
    acceptable names for the sets are: timit61, timit48, timit41, timit39  (upper or lowercase allowed)
    a mapping specification of the form "set1_set2" overrides the set1/set2 definitions
    
    default: "timit61" to "cmu"
    '''

    xlat_dict = None
    if mapping is not None:
        # first 3 are legacy code -- subject to deprecation
        if(mapping == 'timit61_48') : xlat_dict = timit61_48_diff
        elif(mapping == 'timit61_41') : xlat_dict = timit61_41_diff
        elif(mapping == 'timit61_39') : xlat_dict = timit61_39_diff
        else: set1,set2 = mapping.split("_")
        if xlat_dict is not None: return(xlat_dict)
    
    fname = pkg_resources.resource_filename('pyspch', 'data/timit-61-48-39-41.txt')
    timit_map = read_data_file(fname, maxcols = 4, as_cols=True)
    col_map={"timit61":0,"timit48":1,"timit39":2,"timit41":3,"cmu":3}
    
    col_set1 = col_map[set1.lower()]
    col_set2 = col_map[set2.lower()]
    
    timit_map = dict(zip(timit_map[col_set1],timit_map[col_set2]))
    return(timit_map)


####################################################################
# TIMIT CORPORA TOOLS
####################################################################

### Corpus from directory ###

def get_all_files(path):
    
    files = os.listdir(path)
    fnames = list()
    
    for entry in files:
        file = os.path.join(path, entry)
        if os.path.isdir(file):
            fnames = fnames + get_all_files(file)
        else:
            fnames.append(file)
            
    return fnames

def get_corpus(path):
    """
    Returns all files in path (without extensions)
    """    
    # get all filenames
    fnames = get_all_files(path)

    # remove root and extention + to posix + remove duplicates
    fnames = [os.path.relpath(fname, path) for fname in fnames]
    fnames = [Path(fname).as_posix() for fname in fnames]
    fnames = [os.path.splitext(fname)[0] for fname in fnames]
    fnames = list(set(fnames))
    
    return sorted(fnames)

def make_dirs_for_corpus(root_path, rel_path_list):
    for rel_path in rel_path_list:
        os.makedirs(root_path + os.path.join(os.path.dirname(rel_path),''), mode=755, exist_ok=True)

def get_timit_corpus(path, 
        split="(train|test)", 
        region="dr[12345678]", 
        speaker="(m|f)",
        sentence="(si|sx|sa)"):
    """
    Returns TIMIT files in path (without extensions).
    Regular expressions (arguments) rely on TIMIT directory structure 
    and can adapted to obtain a TIMIT subset.
    """
    # get all filenames
    fnames = get_corpus(path)
    
    # regex filtering
    fnames = filter_list_timit(fnames, split, region, speaker, sentence)
    
    return fnames


def get_timit_metadata(fnames):
    """
    Returns DataFrame containing meta data derrived from TIMIT filenames.
    Regular expressions (arguments) rely on TIMIT directory structure to extract meta data.
    """
    # TIMIT metadata (from relative path, with regex)
    rgx_split = re.compile(f'.*(train|test)/.*')
    rgx_region = re.compile(f'.*/(dr\d)/.*')
    rgx_gender = re.compile(f'.*/(m|f).{{4}}/.*')
    rgx_speaker = re.compile(f'.*/([mf].{{4}})/.*')
    rgx_sentence = re.compile(f'.*/(si|sa|sx).*')
    
    # metadata lists
    split = [rgx_split.search(fname).group(1) for fname in fnames]
    region = [rgx_region.search(fname).group(1) for fname in fnames]
    gender = [rgx_gender.search(fname).group(1) for fname in fnames]
    speaker = [rgx_speaker.search(fname).group(1) for fname in fnames]
    sentence = [rgx_sentence.search(fname).group(1) for fname in fnames]
    
    # dataframe
    meta_df = pd.DataFrame([fnames, split, region, gender, speaker, sentence]).T
    
    return meta_df

def filter_list_regex(fnames, rgx):
    # regex filtering
    rgx = re.compile(rgx)
    fnames_filt = [ fname for fname in fnames if rgx.match(fname) ]
    
    return sorted(fnames_filt)
  
### TIMIT corpus ###

def filter_list_timit(fnames, 
        split="(train|test)", 
        region="dr[12345678]", 
        speaker="(m|f)",
        sentence="(si|sx|sa)"):
    
    # regex for TIMIT 
    rgx = f'.*{split}.*/{region}.*/{speaker}.*/{sentence}.*'

    return filter_list_regex(fnames, rgx)






####################################################################
# PROCESSING of (TIMIT like) SEGMENTATION FILES
####################################################################

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
    xlat (str) :  optional phoneme mapping. e.g. "timit61_cmu"
    
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

    if(xlat is None):        
        xlat_dict = None
    elif isinstance(xlat,dict):  
        xlat_dict = xlat
    else:
        xlat_dict = get_timit_mapping(xlat)    

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
    
