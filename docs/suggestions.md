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
