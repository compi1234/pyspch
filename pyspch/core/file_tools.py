import os, sys, io, pkg_resources
import numpy as np
import pandas as pd
import json
from urllib.request import urlopen
from urllib.parse import urlparse
from pathlib import Path
import scipy.io as sio

#from . import audio, timit
    
###########################
# FILE TOOLS
############################

def open_fobj(resource):
    '''
    opens a file like object for reading from either a filename (local or mounted) or a URL resource (https)
    and read all data into a BytesIO object
    '''
    parsed=urlparse(resource)
    if(parsed.scheme == 'https'):
        f = urlopen(resource)
    else:
        f = open(resource,"rb")
    return(f)


def read_fobj(resource):
    '''
    opens a file like object for reading from either a filename (local) or a URL resource
    and reads all data into a BytesIO object
    '''  
    try:
        return(io.BytesIO(open_fobj(resource).read()))
    except:
        return(None)
    
def read_dataframe(resource,sep='\t',names=None,dtype=None,strip=True):
    '''
    reads a column organized datafile into a panda's DataFrame
    This is a wrapper around pd.read_csv()
    
    Arguments:
        sep     separator, default='\t', use '\s+' for whitespace separation
        names   names of the columns (Default is None)
        dtype   dtype conversion to enforce (Default is None)
        strip   Boolean, strip white space at edges of a datafield (Default is True)
    '''
    try:
        df = pd.read_csv(resource,sep=sep,header=None,names=names,dtype=dtype)
        # this will convert object dtypes to string (most of the time)
        df = df.convert_dtypes()
        # optionally strip trailing white spaces on string fields
        if strip:
            df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        return(df)
    except:
        return(None)


def read_data_file(resource,encoding='utf-8',maxcols=None,as_cols=False):
    '''
    print('DEPRECATION WARNING:  - pls. use read_txt() instead')
    '''
    try:
        return(read_txt(resource,encoding=encoding,maxcols=maxcols,as_cols=as_cols))
    except:
        return(None)

def read_txt(resource,encoding='utf-8',maxcols=None,as_cols=False):
    '''
    generic read_data_file() routine with optional encoding setting
    reads lines from local file or URL resource
    
    for local files equivalent to read().splitlines()
    
    Arguments:
        maxcols (int, default=None)  if specified, each line is split on blanks into (maxcols+1) cols
                        the last column may be a string with blacks   
        as_cols (boolean, default= False)  if True, data is returned as list of columns instead of (lines with splits)
                        only applicable if maxcols is not None
    '''
    try:
        # read into lines
        f = read_fobj(resource)
        lines = []
        for line in f:
            line = line.decode(encoding).strip()
            if maxcols is not None:
                line = line.split(None,maxcols)
            lines.append(line)
        f.close()
    
        if( as_cols and (maxcols is not None) ):
            # split into columns
            cols = [ [] for _ in range(maxcols) ]
            for line in lines:
                for c in range(maxcols):
                    cols[c].append(line[c])
            return(cols)
        else:
            return(lines)
    except:
        print(f"WARNING(read_data_file): reading from file {resource} failed")
        return(None)

    
def write_txt(txt, filename, eol='\n'):
    '''
    write_txt() generic write routine for txt data
    
    writes *txt* to a LOCAL FILE 
    txt must be either a single string or a list of strings, in case of list the *eol* character is added to each line
    
    arguments:
        txt       list or str
        
    '''
    if type(txt) is list: 
        txt = [line + eol for line in txt]
    open(filename, 'w').writelines(txt) 
    

def lines_to_columns(lines,maxcols=2,sep=None):
    '''
    rearrange an array of lines holding column data into columns
    
    parsing is done using the separator definitions of the builtin split() function
        - the default sep (separator) is None
        - the maximum number of columns to extract is maxcols (as passed to split())
    '''
    cols = [ [] for _ in range(maxcols) ]
    for line in lines:
        line = line.split(sep,maxcols)        
        for c in range(maxcols):
            cols[c].append(line[c])
    return(cols)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    
def write_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4, cls=NpEncoder)

def read_json(filename):
    with open_fobj(filename) as f:
        data = json.load(f)
    return data
