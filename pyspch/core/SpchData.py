#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import os
import re

import numpy as np
import pandas as pd

# import modules
from . import read_fobj, seg2lbls
from .audio import load
from .timit import read_seg_file
from ..sp import feature_extraction

### SpchData ###

class SpchData(object):
    
    def __init__(self, file_ids):
        # attributes
        self.file_ids = file_ids # list of file_id's
        self.signals = None
        self.features = None
        self.labels = None
        self.n_frames = None
        # write/read
        self.write_fnc = lambda file, array: np.save(file + '.npy', array)
        self.read_fnc = lambda file: np.load(read_fobj(file + '.npy'))  
        
    # General    
    def read(self, path, attr_name=None):
        lst = []
        for file_id in self.file_ids:
            readfile_id = os.path.join(path, file_id)
            arr = self.read_fnc(readfile_id)
            lst.append(arr)
        if attr_name is not None: setattr(self, attr_name, lst)
        else: return lst
    
    def write(self, path, arr):
        for file_id in self.file_ids:
            writefile_id = os.path.join(path, file_id)
            self.write_fnc(writefile_id, arr)

    # Signals
    def read_signals(self, feature_path, sample_rate, extension='.wav'):
        self.signals = []
        for file_id in self.file_ids:
            # read wav file (enforce sample rate)
            wavfile_id = os.path.join(feature_path, file_id + extension)
            wavdata, _ = load(wavfile_id, sample_rate=sample_rate)
            self.signals.append(wavdata)
    
    # Features       
    def extract_features(self, feature_path, feature_args, extension='.wav'):
        self.features = []
        for file_id in self.file_ids:
            # read wav file (enforce sample rate)
            wavfile_id = os.path.join(feature_path, file_id + extension)
            wavdata, _ = load(wavfile_id, sample_rate=feature_args['sample_rate'])
            # extract feature
            feature = feature_extraction(wavdata, **feature_args)
            self.features.append(feature)
    
    def extract_features_from_signals(self, feature_args):
        self.features = []
        for wavdata in self.signals:
            # extract feature
            feature = feature_extraction(wavdata, **feature_args)
            self.features.append(feature)
    
    def modify_features(self, modify_feature_args):
        for i, feature in enumerate(self.features):
            self.features[i] = feature_extraction(spg=feature, **modify_feature_args)
    
    def write_features(self, feature_path):
        for file_id, feature in zip(self.file_ids, self.features):
            featurefile_id = os.path.join(feature_path, file_id)
            self.write_fnc(featurefile_id, feature)  
            
    def read_features(self, feature_path, modify_feature_args={}):
        self.features = []
        for file_id in self.file_ids:
            featurefile_id = os.path.join(feature_path, file_id)
            feature = self.read_fnc(featurefile_id)
            # modify feature on the fly (can save memory)
            if modify_feature_args:
                feature = feature_extraction(spg=feature, **modify_feature_args)
            self.features.append(feature)
                
    # Labels   
    def extract_labels(self, seg_path, shift, extension='.phn'):
        self.labels = [] 
        for file_id in self.file_ids:
            # read segmentation 
            segfile_id = os.path.join(seg_path, file_id + extension)
            seg_df = read_seg_file(segfile_id)
            # extract labels
            label = seg2lbls(seg_df, shift)
            self.labels.append(np.array(label))
    
    def extract_alligned_labels(self, seg_path, shift, pad_lbl='', extension='.phn'):
        self.labels = [] 
        if self.features and self.n_frames:
            self.n_frames = self.get_nframes('features')
        else:
            print("First set self.features or self.n_frames")
            
        for file_id, n_frames in zip(self.file_ids, self.n_frames):
            # read segmentation 
            segfile_id = os.path.join(seg_path, file_id + extension)
            seg_df = read_seg_file(segfile_id)
            # extract labels
            label = seg2lbls(seg_df, shift, n_frames=n_frames, end_time=None, pad_lbl=pad_lbl)
            self.labels.append(np.array(label))       
 
    def extract_labels_from_meta(self, meta: pd.DataFrame, col_file_id=0, col_label=-1):
        # filter meta data with file_ids
        meta_filtered = meta[meta[col_file_id].isin(self.file_ids)]
        meta_labels = meta_filtered[col_label]
        # extract labels
        self.labels = []
        if self.features and self.n_frames:
            self.n_frames = self.get_nframes('features')
        else:
            print("First set self.features or self.n_frames")
        for meta_label, n_frames in zip(meta_labels, self.n_frames):
            label = [meta_label] * n_frames
            self.labels.append(np.array(label))
    
    def modify_labels(self, lab2lab_dct):
        for i, label in enumerate(self.labels):
            self.labels[i] = [lab2lab_dct[lab] for lab in self.labels]   
            
    def write_labels(self, seg_path):
        for file_id, seg in zip(self.file_ids, self.labels):
            segfile_id = os.path.join(seg_path, file_id)
            self.write_fnc(segfile_id, seg)  
        
    def read_labels(self, seg_path):
        self.labels = []
        for file_id in self.file_ids:
            segfile_id = os.path.join(seg_path, file_id)
            feature = self.read_fnc(segfile_id)
            self.labels.append(feature)
    
    # Filter  
    def filter(self, bool_filter, inplace=False):
        # inplace or new SpchData object
        if inplace: 
            new = self
        else: 
            new = SpchData(None)
        # filter attributes
        for name in ['file_ids', 'signals', 'features', 'labels']:
            attr = getattr(self, name)
            if attr is not None:
                new_attr = [i for (i, match) in zip(attr, bool_filter) if match]
                setattr(new, name, new_attr)
        if not inplace: 
            return new
              
    def subset(self, subset, inplace=False):
        filt = [ True if file_id in subset else False for file_id in self.file_ids ]
        return self.filter(filt, inplace) 
    
    def subset_with_regex(self, rgx, inplace=False):
        rgx = re.compile(rgx)
        filt = [ True if rgx.match(file_id) else False for file_id in self.file_ids ]
        return self.filter(filt, inplace)
       
    # get from attribute
    def get_nframes(self, name, axis=-1):
        attr = getattr(self, name)
        if attr is None: return []
        else: return [item.shape[axis] for item in attr]
    
    def get_set(self, name='labels'):
        attr = getattr(self, name)
        return set(itertools.chain.from_iterable(attr))
    
    def get_features_as_numpy(self):
        return np.hstack(self.features).T

    def get_labels_as_numpy(self):
        return np.hstack(self.labels)
         
    # Dataframe
    def to_dataframe(self, attributes=['file_ids', 'features', 'labels']):
        df_dict = {}
        for attr in attributes:
            df_dict[attr] = getattr(self, attr)
            
        return pd.DataFrame(df_dict)
    
    
        
def DataFrame_to_SpchData(df, delete_df=True, attributes=['file_ids', 'features', 'labels']):
    # initialize with file_ids
    file_ids = df['file_ids'].to_list()
    spchdata = SpchData(file_ids)
    # other attributes
    attributes = [attr for attr in attributes if attr in df.columns]
    for attr in attributes:
        setattr(spchdata, attr, df[attr].to_list())
        if delete_df: df.drop(attr, axis=1, inplace=True)
    if delete_df: del df
    
    return spchdata
