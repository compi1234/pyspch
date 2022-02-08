# New Organization with Subpackages

subpackages:  (all modules in a subpackage are fully loaded on init )
- io
- utils
- sp
- display




### Signal Processing

- cepstrum(wavdata=None, Spg=None, n_mel=None, n_cep=None )
    - input: waveform or spectrogram data
    - n_mel: if None regular cepstrum, if mel
    - n_cep: number of cepstral coefficients to be returned (if None = all, full resolution)

- spg2cep(spgdata, n_cep=None)
        
- cep2spec()
    
    
Data objects

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
