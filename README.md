# pyspch


### Notes on v0.6 

- major reorganization with subpackages
    - core : for (file) I/O related stuff and general utilities; all these routines are included in the main import
    - sp : subpackage that can be viewed as an extension to and built upon librosa, with some specific speech signal processing routines (though single channel mainly) 
            now includes, cepstral processing, feature manipulation, time domain feature extraction, ..
    - display : all display routines
    - nn :   a neural net subpackage built on torch

- default imports
compatibility stays assured for python>=3.7, version numbers were raised for a few key packages:
    + matplotlib>=3.4
    + librosa>=0.9

- Colab:
On installing in colab a few warnings that come from matplotlib (higher version than default 3.2.2).
From experience we know that these warnings can be neglected; though it may be best (required) to restart the runtime.




### Notes on v0.5 -- final version for H02A6 course in 2021

bumped a few minimal requirements, especially
pandas > 1.1      to avoid read_pickle() problems
                  while this function isn't used inside, it seems a good idea 

and for the HMM part, usage of scikit-learn >= 1.0.1 is required, also implying 
Python (>= 3.7)
NumPy (>= 1.14.6)
SciPy (>= 1.1.0)
joblib (>= 0.11)
threadpoolctl (>= 2.0.0)


### Notes on v0.4 -- 29/09/2021

- this should become the first more or less stable version to be used during the academic year 2021-2022
- there was some restructuring in the sense that tests and demos have been moved out of the package, but are still in the project one level up
- some of the demos - linked closely to my Speech Recognition course (H02A6) have been moved into the independent repository 'spchlab' which is linked to that course

 
### Notes on v.03

- there were many breaking changes in going from 01->02->03
- the plotly backend is temporarily not supported

### Known Issues with Dependencies

- librosa  does not seem to work the latest versions of numba, llvmlite
- numba <= 0.50 (don't know if this is the highest version, but 0.53 definitely seems incompatible with librosa)
- llvmlite <= 0.33 (automatically with numba <= 0.50)

### Known Issues in v0.1

- soundfile and librosa are not supported on binder, but critical for all I/O imports
    - need to find a way to load system library libsndfile on the lINUX servers ( maybe: sudo apt-get install libsndfile1 ; not tested yet )
- multichannel recording in Colab not supported
- discrepancies between mpl and plotly low level API for axis selection
