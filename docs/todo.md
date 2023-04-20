# BUGS

- dtw module
    - name changes of dtw.lev_distance and dtw.edit_distance not implemented in score_corpus
    - names of routines to be reevaluated
    - holistic approach for both discrete and continuous inputs

# TODO 

sanity check on discrete probabilities: use of labels vs. use of indices
- evolving standard in sklearn
- could be with index=True/False as option
- now most work with indices only ...

package version updates v0.8 ? (yml vs colab)
- soundfile 0.10 - 0.11
- torchaudio 0.13 / torch 1.13 / cuda 11.6(7) ?
- ipywidgets 8.0 - 7.7
- graphviz 0.16 - 0.10

sequence data examples
- use scipy.vq.vq for discretization
- eliminate usage of iris dataset

check multiple occurence of
- confusion_matrix
- mean_variance-Norm




# Data objects

it is worth considering putting all signal data (waveforms, spectrograms) in a data object 

### waveform data object
data            float32 - wavdata
n_samples       int
n_channels      int
sample_rate     int
dx              float
x_label         str ("Time(sec)")
y_label         str

### spectrogram data object
data            float32 - spgdata
n_frames        int
n_param         int
dx              float
dy              float
x_label         str ("Time")
y_label         str ("Frequency(Hz)")

### segmentation data object 
is a panda's dataframe



# Code Modifications

1. make the frames[] parameters a slice object
2. put spectrograms also in a (n_frames,n_param) array (transposed vs now)
3. streamline n_xxx vs nxxx


## Converting data to (nframes,nfeatures) ??

- most routines in 'machine learning' have 'nsamples' (or 'nframes') as first argument
- signal processing and display routines have the opposite

### work to be done to stream line this ?
! generally search for reshape statements

- Densities.py, - libhmm.py, - GaussianMixtureClf.py, dtw.py
    + OK for 2D structures
    + reconsider default inference for 1D inputs: from (n_features,) to (n_samples,) in
        - lbl2indx
        - predict_ftr_probz
        
- sp , core.audio
    + leave this intact as intended as 'librosa'-compatible package , and just change all the rest ?
    

- display
    + consistent with sp ? or machine learning ?
    + add_line_plot
    
