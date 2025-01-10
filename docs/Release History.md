# pyspch

### TBD / work for future releases
- package_resources will get deprecated with Python 3.11, but the usage of importlib.resources was not stable between 3.7 and 3.10 so it needs to be discouraged for the time being
- Deprecation warning ahead of pandas 3.0: *C:\Users\compi\AppData\Local\Temp\ipykernel_20852\3573829142.py:6: DeprecationWarning: 
Pyarrow will become a required dependency of pandas in the next major release of pandas (pandas 3.0),*
- pickled data files in sklearn may get unusable with future sklearn releases, usage needs to be reassessed
- utils_timit should be integrated core.timit
- utils_clf   should be integrated in stats subpackage

### Notes on v0.8.3
- modifications on iSpectrogram
  +   added spec envelope and residue
  +   freq labels and axis in RHS  (but still missing for some in LHS plots)
  +   merged into main 21/02/2024; saved as v0.8.3b
 - 23/02/2024: enforced consistency on mel_defaults throughout Sps
 - 06/03/2024: add loading of pickle files to load_data()
 - 18/03/2024:  utils_clf and utils_timit were added to core
               this is not a good permanent solution
 - 20/03/2024: add_line_plot() - color cycling is restored as default
 - 24/03/2024: small bug fixes in libhmm and probdist
 - 18/04/2024: added utils_x to core
 - 22/05/2024: make logprob_foor dominate prob_floor
 - 22/05/2024: all probs are float64
 - 29/05/2024: added .draw() method to hmm
 - 21/08/2024: added arguments i1 and i2 to .plot_trellis() for partial plots 
 
### Notes on v0.8.2
- added the mel.py module in sp
- added arguments (segwav) to PltSpgFtrs()
- last loaded to main 31/01/2024
   
- the iSpectrogram() interactive spectrogram is reworked considerably
- restructuring of the ./data directory, now structured with subdirs
- a number of smaller bug fixes/patches
- frozen on 29/01/2024

### Notes on v08.01
- This is an intermediate work-in-progress release (with inconsistent naming) frozen on 16/01/2024

### Notes on v0.8

- this is a release with major breaking changes
- support for Python 3.7 is dropped
- Python 3.10 has become the reference Python release
- the main stack consists of numpy=1.22, scikit-learn=1.2, pandas=1.5, librosa=0.10, matplotlib=3.7
- torchaudio=2.10+cuda=11.8
- consolidation of test suites in the ./tests directory
  
### Notes on v07.03

- now supports tied observations
    + implemented via defining an obs_indx array which specifies the observation index to use in each state
    + should be transparent for all old code as the above defaults to a 1-1 mapping

### Notes on v0.7

- recommended Python versions are 3.7, 3.8 and 3.9
    + packages have been upgraded to latest available versions under 3.7
    + for compatibility with COLAB, a few packages were downgraded to a lower version (we shouldn't loose much)
        + matplotlib  3.2.*
        + librosa     0.8.*
- COLAB's major inconsistencies with playing AUDIO are now gone as they moved to IPython 7.x

- torch and torchaudio are recommended, but not required for most of the package except the nn subpackage
    + installation is not included in the reference .yml file as things may not be perfectly coded for everyone
    + example installation when using conda (do this after creating your env
        > conda install pytorch torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge 



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
