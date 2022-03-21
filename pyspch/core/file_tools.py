import os, sys, io
import numpy as np
import pandas as pd
from urllib.request import urlopen
from urllib.parse import urlparse
from pathlib import Path


    
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
    return(io.BytesIO(open_fobj(resource).read()))

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
    df = pd.read_csv(resource,sep=sep,header=None,names=names,dtype=dtype)
    # this will convert object dtypes to string (most of the time)
    df = df.convert_dtypes()
    # optionally strip trailing white spaces on string fields
    if strip:
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    return(df)


def read_data_file(resource,encoding='utf-8',maxcols=None,as_cols=False):
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