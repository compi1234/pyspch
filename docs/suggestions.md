# TODO 

sequence data examples
- use scipy.vq.vq for discretization
- eliminate usage of iris dataset

check multiple occurence of
- confusion_matrix
- mean_variance-Norm

Ipython.display : display, clear_output  seem obsolete in newest version   

update referrals to Colab as colab has moved to python 3.7 and Ipython

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
