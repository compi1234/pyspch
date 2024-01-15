# pyspch: speech analysis and recognition package for teaching

## What is it ?
**pyspch** is a Python package that provides easy to use speech analysis and speech recognition functionality in Python.  

The focus is on EASE of USE and SIMPLE ACCESS to the science and algorithms behind speech processing today.   This is by no means intended to be a state-of-the-art package for building speech recognition systems.

The principle usage of **pyspch** is with *Jupyter Notebooks* in which small systems or demonstrators are built, such as  the suite of demonstrations and exercises in the **spchlab** repository( https://github.com/compi1234/spchlab)


## Main features
 
- Jupyter Notebook toolkit for speech visualization and audio I/O
- speech analysis:
    + feature extraction: MEL and CEP
    + postprocessing and grouping
    + spectrograms (Fourier, MEL and CEP)
- speech recognition:
    + dynamic time warping
    + Bayesian classifiers
    + Hidden Markov Models
    + Deep Neural Nets (Multilayer Perceptrons)


## Installation

The source code resides on GitHub at:
https://github.com/compi1234/pyspch

> pip install git+https://github.com/compi1234/pyspch.git

## Dependencies

- recommended Python versions are 3.10 or 3.9
- torch and torchaudio are recommended, but not required for most of the package except the nn subpackage (DNNs)
    + installing with conda on Windows (do this after creating your normal env)
        > conda install pytorch==2.1.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
    + You may find a more compatible version for your system on the [pytorch website]( https://pytorch.org/get-started/previous-versions/)
