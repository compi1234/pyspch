#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ast import Str
import os, re
from os.path import relpath, splitext
from pathlib import Path

import numpy as np
import pandas as pd
# import h5py


# import modules
import pyspch.sp as Sps
import pyspch.core as Spch
import nn.utils as utils

### Corpus from directory ###

def get_corpus(path):
    """
    Returns all files in path (without extensions)
    """    
    # get all filenames
    fnames = utils.get_all_files(path)

    # remove root and extention + to posix + remove duplicates
    fnames = [relpath(fname, path) for fname in fnames]
    fnames = [Path(fname).as_posix() for fname in fnames]
    fnames = [splitext(fname)[0] for fname in fnames]
    fnames = list(set(fnames))
    
    return sorted(fnames)

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

def get_corpus_timit(path, 
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
    
### Read/write array data ###

class ArrayWriter(object):
    
    def __init__(self, mode='numpy', extension='.npy'):
        self.mode = mode
        self.extension = extension
        self.set_write_fnc()
    
    def set_write_fnc(self, write_fnc=None):
        # default 
        if write_fnc is None:
            if self.mode == 'numpy':
                write_fnc = lambda file, array: np.save(file, array) 
            # elif self.mode == 'hdf5':
            #     write_fnc = lambda file, array: h5py.File(file, 'w').create_dataset('data', data=array)
            else:
                write_fnc = lambda file, array: open(file, 'w').write(array)
        # set
        self.write_fnc = write_fnc
        
    def write(self, file, array, extension=None):
        if extension is None: extension = self.extension
        file = utils.format_extension(file, extension)
        utils.make_dir_for_file(file)
        self.write_fnc(file, array)                

class ArrayReader(object):
    
    def __init__(self, mode='numpy', extension='.npy'):
        self.mode = mode
        self.extension = extension
        self.set_read_fnc()
    
    def set_read_fnc(self, read_fnc=None):
        # default 
        if read_fnc is None:
            if self.mode == 'numpy':
                read_fnc = lambda file: np.load(file) 
            # elif self.mode == 'hdf5':
            #     read_fnc = lambda file: h5py.File(file, 'r').get('data')
            else:
                read_fnc = lambda file: open(file, 'w').read()
        # set
        self.read_fnc = read_fnc
        
    def read(self, file, extension=None):
        if extension is None: extension = self.extension
        file = utils.format_extension(file, extension)
        return self.read_fnc(file)   

### Labels ###

def seg2label(phnseg, sample_rate, f_shift):
    labels = []
    for _, seg in phnseg.iterrows():
        f0 =  int(seg['t0'] / (f_shift * sample_rate))
        f1 =  int(seg['t1'] / (f_shift * sample_rate))
        labels += [seg['seg']] * int(f1 - f0)
        
    return labels 

### SpchData ###

class SpchData(object):
    
    def __init__(self, corpus):
        self.corpus = corpus
        self.signals = None
        self.features = None
        self.feature_args = None
        self.labels = None
        # TODO: include meta data?
    
    ### Signals (raw)
    
    def read_signal(self, feature_path, sample_rate, extension='.wav'):

        # extract features per file in corpus  
        self.signals = []
        for fname in self.corpus:
            # file paths
            wavfname = os.path.join(feature_path, fname + extension)
            # read wav file (enforce sample rate)
            signal, _ = Spch.load(wavfname, sample_rate=sample_rate)
            self.signals.append(signal)

    ### Features
       
    def extract_features(self, feature_path, feature_args, extension='.wav'):
        self.feature_args = feature_args
        # extract features per file in corpus  
        self.features = []
        for fname in self.corpus:
            # file paths
            wavfname = os.path.join(feature_path, fname + extension)
            # read wav file (enforce sample rate)
            wavdata, _ = Spch.load(wavfname, sample_rate=feature_args['sample_rate'])
            # extract and splice feature
            feature = Sps.feature_extraction(wavdata, **feature_args)
            self.features.append(feature)
    
    def pad_features(self, lengths, dim=-1):
        
        # pad/trim labels until feature lengths match given lengths 
        features_padded = []
        for feature, length in zip(self.features, lengths):
            feature_padded = enforce_length_array(feature, length, dim)
            features_padded.append(feature_padded)
        
        self.features = features_padded
    
    def write_features(self, feature_path, writer: ArrayWriter):

        for fname, feature in zip(self.corpus, self.features):
            # file paths
            featurefname = os.path.join(feature_path, fname)
            # write feature file
            writer.write(featurefname, feature)  
            
    def read_features(self, feature_path, reader: ArrayReader):
        
        # read features per file in corpus  
        self.features = []
        for fname in self.corpus:
            # file paths
            featurefname = os.path.join(feature_path, fname)
            # read feature file
            feature = reader.read(featurefname)
            self.features.append(feature)
        
    ### Labels
    
    def extract_labels(self, seg_path, seg_args, extension='.phn'):
        
        # extract segmentation per file in corpus  
        self.labels = [] 
        for fname in self.corpus:
            # file paths
            segfname = os.path.join(seg_path, fname + extension)
            # get segmentation 
            seg_df = Spch.read_seg_file(segfname)
            label = seg2label(seg_df, seg_args['sample_rate'], seg_args['f_shift'])
            self.labels.append(np.array(label))
 
    def pad_labels(self, lengths, dim=-1):
        
        # pad/trim labels until label lengths match given lengths 
        labels_padded = []
        for label, length in zip(self.labels, lengths):
            label_padded = enforce_length_array(label, length, dim)
            labels_padded.append(label_padded)
        
        self.labels = labels_padded
    
    def pad_labels_with_token(self, lengths, pad_token, dim=-1):
        
        # pad/trim labels until label lengths match given lengths 
        labels_padded = []
        for label, length in zip(self.labels, lengths):
            dlength = length - label.shape[dim]
            label_padded = np.concatenate((label, np.array([pad_token] * dlength))) if dlength >= 0 else label[:dlength]
            labels_padded.append(np.array(label_padded))
        
        self.labels = labels_padded
             
    def extract_labels_from_meta(self, meta: pd.DataFrame, lengths: list, col_fname=0, col_label=-1):
        
        # filter meta data and make dictionairy
        meta_filtered = meta[meta[col_fname].isin(self.corpus)]
        meta_labels = meta_filtered[col_label]
        assert len(self.corpus) == len(meta_labels), "corpus contains fnames not found in meta data"
        
        # extract labels
        self.labels = [] 
        for meta_label, length in zip(meta_labels, lengths):
            label = [meta_label] * length
            self.labels.append(np.array(label))
               
    def write_labels(self, seg_path, writer: ArrayWriter):
        
        for fname, seg in zip(self.corpus, self.labels):
            # file paths
            segfname = os.path.join(seg_path, fname)
            # write feature file
            writer.write(segfname, seg)  
        
    def read_labels(self, seg_path, reader: ArrayReader):
        
        # read features per file in corpus  
        self.labels = []
        for fname in self.corpus:
            # file paths
            segfname = os.path.join(seg_path, fname)
            # read feature file
            feature = reader.read(segfname)
            self.labels.append(feature)
    
    ### Filter corpus
    
    def filter_with_regex(self, rgx):
        
        # regex filter 
        rgx = re.compile(rgx)
        rgx_filter = [ True if rgx.match(fname) else False for fname in self.corpus ]
        
        # update attributes
        self.corpus = [i for (i, match) in zip(self.corpus, rgx_filter) if match]
        if self.signals is not None:
            self.signals = [i for (i, match) in zip(self.signals, rgx_filter) if match]
        if self.features is not None:
            self.features = [i for (i, match) in zip(self.features, rgx_filter) if match]
        if self.labels is not None:
            self.labels = [i for (i, match) in zip(self.labels, rgx_filter) if match]
    
    def subset_with_regex(self, rgx):
        
        # regex filter 
        rgx = re.compile(rgx)
        rgx_filter = [ True if rgx.match(fname) else False for fname in self.corpus ]
        
        # new SpchData object
        new = SpchData([i for (i, match) in zip(self.corpus, rgx_filter) if match])
        if self.signals is not None:
            new.signals = [i for (i, match) in zip(self.signals, rgx_filter) if match]
        if self.features is not None:
            new.features = [i for (i, match) in zip(self.features, rgx_filter) if match]
        if self.labels is not None:
            new.labels = [i for (i, match) in zip(self.labels, rgx_filter) if match]
            
        return new
    
    ### Length
       
    def get_length_features(self, dim=-1):
        if self.features is not None:
            return [feature.shape[dim] for feature in self.features]
        else: 
            return []
        
    def get_length_labels(self, dim=-1):
        if self.labels is not None: 
            return [label.shape[dim] for label in self.labels]
        else: 
            return []

### Modify array lengths (padding/trimming)

def enforce_length_array(arr, length, dim=-1):
    dlength = length - arr.shape[dim]
    if dlength >= 0:
        arr = np.concatenate(
            (arr, np.repeat(np.array(arr.take(-1, dim))[None,...], dlength, axis=0).T),
            axis=dim)
    else:
        arr = np.delete(arr, np.s_[dlength:], dim)  
    return arr
    
def enforce_lengths(list_of_arrays, lengths, dim=-1):
    
    # pad/trim arrays until array lengths match given lengths 
    list_of_arrays_padded = []
    for arr, length in zip(list_of_arrays, lengths):
        arr = enforce_length_array(arr, length, dim)
        list_of_arrays_padded.append(arr)
    
    return list_of_arrays_padded

### Numpy padding alternative

def format_pad_width_to_ndim(pad_width, ndims, dim=-1):
    pad_widths = list(ndims * ((0, 0),))
    pad_widths[dim] = pad_width
    
    return tuple(pad_widths)

def pad_array_along_axis(arr, pad_width, dim=-1, pad_args={'mode': 'constant'}):
    pad_widths = format_pad_width_to_ndim(pad_width, arr.ndim, dim)
    arr = np.pad(arr, pad_widths, **pad_args)
    
    return arr
        
