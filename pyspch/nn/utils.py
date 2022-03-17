#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd

### Path

def format_extension(file, extension):
        return Path(os.path.splitext(file)[0] + extension).as_posix()

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

def make_dir(path):
    if not os.path.exists(path): 
        os.makedirs(path)

def make_dir_for_file(filepath):
    dirpath = os.path.dirname(filepath)
    make_dir(dirpath)
    
### Read / Write

def read_txt(file, eol='\n'):
    return [line.strip() for line in open(file).readlines()] 

def write_txt(txt, file, eol='\n'):
    if type(txt) is list: 
        txt = [line + eol for line in txt]
    open(file, 'w').writelines(txt) 

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
        json.dump(data, f, cls=NpEncoder)

def read_json(filename):
    with open(filename) as f:
        data = json.load(f)
    return data

### Dataframe 

def read_file_as_df(file, ncols=1, sep=" |\t", enc="utf-8", maxchars=10**6):
    """
    Read any file as dataframe
    """
    df = pd.read_fwf(file, header=None, encoding=enc, widths=[maxchars])
    df = df[0].str.split(sep, n=ncols-1, expand=True)
    
    return df