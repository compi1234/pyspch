import os, sys, io, pkg_resources
import numpy as np
import pandas as pd
import json
from urllib.request import urlopen
from urllib.parse import urlparse
from pathlib import Path
import scipy.io as sio

from . import audio, timit, file_tools
from .hillenbrand import fetch_hillenbrand
from .sequence_data import make_seq1, make_seq1d

def get_full_filename(fname,root=None):
    if root is None: root = "pkg_resources_data"
    if root == "pkg_resources_data":
        filename = pkg_resources.resource_filename('pyspch',"data/"+fname)
    else:
        filename =  root + fname  # maybe this is better: os.path.join(root,fname)
    return(filename)
 
def load_data(name,root=None,**kwargs):
    '''
    A high level generic data loading function.
    Data can be loaded from the example data included in the package as well from directories on disk or URL based resources.

    The precise data loading and processing depends on the filename extension. 
    Extra **kwargs can be passed to the underlying functions.
    The extension based processing is heuristic and therefore only suitable for standardized resources.  
    
    More low level access needs to be used for other resources or resources with names not fitting the standardized format.
    
    Arguments:
        name       (str) filename of resource with extension or named resource
        root       (str) directory (local or https, default is None 
                    the default None will refer to the data directory in the pyspch package
                    only files in subdirs should be accessed
        **kwargs   (dict) passed to called reading routine
        
    Processing in function of extension:
    .wav
            returns waveform data
    .gra, .phn, .syl, .wrd, .seg      
        reads a TIMIT style segmentation file
        returns a segmentation dataframe
    .csv
        reads a comma separated database
        returns the database as DataFrame
    .mat
        reads a MATLAB data file
        returns a dictionary of MATLAB variables
    .lst, .txt
        reads a text files
        returns data as 
            - list of lines
            - dataframe iff 'sep' is specifined as kwarg

    
    '''

    filename = get_full_filename(name,root=root)
    _, ext = os.path.splitext(filename)
    
    if ext == '.wav':        
        wavdata,sample_rate= audio.load(filename,**kwargs)
        return wavdata, sample_rate
    elif ext in ['.gra','.phn','.syl','.wrd','.seg']:
        data = timit.read_seg_file(filename,**kwargs)
        return data
    elif ext in ['.csv']: 
        data = pd.read_csv(filename,**kwargs)
        return data
    elif ext == '.mat':
        raw_data = file_tools.read_fobj(filename)
        data = sio.loadmat(raw_data,squeeze_me=True)
        # remove keys of the ext __xxxxx__
        remove_keys = [ k for k in data.keys() if (k[0]=='_' and k[-1]=='_')]
        for k in remove_keys:    del data[k]
        return data
    elif ext in ['.lst','.txt']:
        if 'sep' in kwargs.keys():
            #print('read dataframe')
            data = file_tools.read_dataframe(filename,**kwargs)
        else:
            #print('read txt')
            data = file_tools.read_txt(filename,**kwargs)
        return data
    elif ext == '':
        if name=='sequence1':
            data= make_seq1(**kwargs)
        elif name =='sequence1d':
            data =  make_seq1d(**kwargs)
        elif name == 'hillenbrand':
            data =  fetch_hillenbrand(**kwargs) 
        elif name == 'tinytimit':
            # returning in ARPABET notation 'aa','iy','uw' instead of stored 'a','i','uw'
            tinytimit =  'https://homes.esat.kuleuven.be/~spchlab/data/timit/features/tinytimit/'
            data_raw = load_data('a-i-uw-800.mat', root=tinytimit)
            data = data_raw
            norm_fac = 40.0
            X_train=data_raw['ALLtrain'].T - norm_fac
            X_test=data_raw['ALLtest'].T - norm_fac
            y_train =np.full((2400,),'aa',dtype='<U2')
            y_train[800:1600] =np.full((800,),'iy',dtype='<U2')
            y_train[1600:2400] =np.full((800,),'uw',dtype='<U2')
            y_test =np.full((600,),'aa',dtype='<U2')
            y_test[200:400] =np.full((200,),'iy',dtype='<U2')
            y_test[400:600] =np.full((200,),'uw',dtype='<U2')
            data = (X_train, X_test, y_train, y_test)
        return data
    
