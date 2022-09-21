# New Organization with Subpackages

subpackages:  (all modules in a subpackage are fully loaded on init )
- core        collects the older subpackages io, audio, utils
- sp          signal processing routines
- display     display routines



### Core Utilities
```
audio.load()       loading of audio file
audio.record()     recording audio
audio.play()       playing audio
audio.stop()       interrupt audio recording

read_data_file()   reads from txt file  (returns 'lines' or 'columns')
                    - from local file or URL resource
read_seg_file()    reads segmentation files (returns segment dataframe)
                    - from local file or URL resource
                    - conversion of time stamps
                    - translation between various TIMIT alphabets

logf()             log with EPS flooring
log10f()           log10 with EPS flooring
check_colab() 
```


**Defined Constants**
```
LOG10       log(10)
LOG2DB      10.0/LOG10
EPS_FLOAT   1.e-39             # used for log-prob flooring
SIGEPS_FLOAT 1.19209290e-7     # Kaldi's flooring on signal energy
SIGEPS_DB    = -69.23689       # Kaldi's energy flooring in dB 
```


**Convenience Functions examplar data functions**
```
fetch_hillenbrand(), select_hillenbrand()  for fetching and selecting data from H-dataset
make_seq1(), make_seq1d()  utilities to generate small set of  sequence data, continuous or discrete
```

### Signal Processing and Feature Extraction  Routines
```
spectrogram()       Fourier or mel spectrogram
spg2mel()           converts Fourier Spectrogram to mel scale
cepstrum()          waveform or spectrogram to cepstrum
cep_lifter()        cepstrum inverse
melcepstrum()

mean_norm()           mean normalization, utterance based
deltas()              delta computation / augmentation
splice_frames()       splicing of frames
```
    

