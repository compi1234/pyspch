#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ast import Str
import os, re
from os.path import relpath, splitext
from pathlib import Path

import numpy as np
import pandas as pd
import itertools

# import modules
from ..core import read_fobj, seg2lbls
from ..core.audio import load
from ..core.timit import read_seg_file
from ..sp import feature_extraction

### SpchData ###

class SpchData(object):
    
    def __init__(self, corpus):
        # attributes
        self.corpus = corpus
        self.signals = None
        self.features = None
        self.labels = None
        self.lengths = None
        # write/read
        self.write_fnc = lambda file, array: np.save(file + '.npy', array)
        self.read_fnc = lambda file: np.load(read_fobj(file + '.npy'))       

    # General    
    def read(self, path, attr_name=None):
        lst = []
        for fname in self.corpus:
            readfname = os.path.join(path, fname)
            arr = self.read_fnc(readfname)
            lst.append(arr)
        if attr_name is not None: setattr(self, attr_name, lst)
        else: return lst
    
    def write(self, path, arr):
        for fname in self.corpus:
            writefname = os.path.join(path, fname)
            self.write_fnc(writefname, arr)

    # Signals
    def read_signals(self, feature_path, sample_rate, extension='.wav'):
        self.signals = []
        for fname in self.corpus:
            # read wav file (enforce sample rate)
            wavfname = os.path.join(feature_path, fname + extension)
            wavdata, _ = load(wavfname, sample_rate=sample_rate)
            self.signals.append(wavdata)
    
    # Features       
    def extract_features(self, feature_path, feature_args, extension='.wav'):
        self.features = []
        for fname in self.corpus:
            # read wav file (enforce sample rate)
            wavfname = os.path.join(feature_path, fname + extension)
            wavdata, _ = load(wavfname, sample_rate=feature_args['sample_rate'])
            # extract feature
            feature = feature_extraction(wavdata, **feature_args)
            self.features.append(feature)
    
    def extract_features_from_signals(self, feature_args):
        self.features = []
        for wavdata in self.signals:
            # extract feature
            feature = feature_extraction(wavdata, **feature_args)
            self.features.append(feature)
    
    def write_features(self, feature_path):
        for fname, feature in zip(self.corpus, self.features):
            featurefname = os.path.join(feature_path, fname)
            self.write_fnc(featurefname, feature)  
            
    def read_features(self, feature_path, modify_feature_args=None):
        self.features = []
        for fname in self.corpus:
            featurefname = os.path.join(feature_path, fname)
            feature = self.read_fnc(featurefname)
            # modify feature on the fly (can save memory)
            if modify_feature_args is not None:
                feature = feature_extraction(feature, **modify_feature_args)
            self.features.append(feature)
                
    # Labels   
    def extract_labels(self, seg_path, shift, extension='.phn'):
        self.labels = [] 
        for fname in self.corpus:
            # read segmentation 
            segfname = os.path.join(seg_path, fname + extension)
            seg_df = read_seg_file(segfname)
            # extract labels
            label = seg2lbls(seg_df, shift)
            self.labels.append(np.array(label))
    
    def extract_alligned_labels(self, seg_path, shift, pad_lbl='', extension='.phn'):
        self.labels = [] 
        if self.lengths is None and self.features is None:
            print("Set self.lengths first")
        elif self.lengths is None:
            self.lengths = self.get_length('features')
        for fname, length in zip(self.corpus, self.lengths):
            # read segmentation 
            segfname = os.path.join(seg_path, fname + extension)
            seg_df = read_seg_file(segfname)
            # extract labels
            label = seg2lbls(seg_df, shift, n_frames=length, end_time=None, pad_lbl=pad_lbl)
            self.labels.append(np.array(label))       
 
    def extract_labels_from_meta(self, meta: pd.DataFrame, col_fname=0, col_label=-1):
        # filter meta data with corpus
        meta_filtered = meta[meta[col_fname].isin(self.corpus)]
        meta_labels = meta_filtered[col_label]
        # extract labels
        self.labels = [] 
        if self.lengths is None and self.features is None:
            print("Set self.lengths first")
        elif self.lengths is None:
            self.lengths = self.get_length('features')
        for meta_label, length in zip(meta_labels, self.lengths):
            label = [meta_label] * length
            self.labels.append(np.array(label))
               
    def write_labels(self, seg_path):
        for fname, seg in zip(self.corpus, self.labels):
            segfname = os.path.join(seg_path, fname)
            self.write_fnc(segfname, seg)  
        
    def read_labels(self, seg_path):
        self.labels = []
        for fname in self.corpus:
            segfname = os.path.join(seg_path, fname)
            feature = self.read_fnc(segfname)
            self.labels.append(feature)
    
    # Filter  
    def filter(self, bool_filter, inplace=False):
        # inplace or new SpchData object
        if inplace: 
            new = self
        else: 
            new = SpchData(self.corpus)
        # filter attributes
        for name in ['corpus', 'signals', 'features', 'labels']:
            attr = getattr(self, name)
            if attr is not None:
                new_attr = [i for (i, match) in zip(attr, bool_filter) if match]
                setattr(new, name, new_attr)
        if not inplace: 
            return new
              
    def subset(self, subset, inplace=False):
        filt = [ True if fname in subset else False for fname in self.corpus ]
        return self.filter(filt, inplace) 
    
    def subset_with_regex(self, rgx, inplace=False):
        rgx = re.compile(rgx)
        filt = [ True if rgx.match(fname) else False for fname in self.corpus ]
        return self.filter(filt, inplace)
       
    # get from attribute
    def get_length(self, name, axis=-1):
        attr = getattr(self, name)
        if attr is None: return []
        else: return [item.shape[axis] for item in attr]
    
    def get_set(self, name='labels'):
        attr = getattr(self, name)
        return set(itertools.chain.from_iterable(attr))
    
      
    # Dataframe
    def to_dataframe(self):
        df_dict = {}
        attributes = ['corpus', 'signals', 'features', 'labels']
        for attr in attributes:
            df_dict[attr] = getattr(self, attr)
            
        return pd.DataFrame(df_dict)
        
def DataFrame_to_SpchData(df, delete_df=True):
    # initialize with corpus
    corpus = df['corpus'].to_list()
    spchdata = SpchData(corpus)
    # other attributes
    attributes = ['corpus', 'signals', 'features', 'labels']
    attributes = [attr for attr in attributes if attr in df.columns]
    for attr in attributes:
        setattr(spchdata, attr, df[attr].to_list())
        if delete_df: df.drop(attr, axis=1, inplace=True)
    if delete_df: del df
    
    return spchdata
