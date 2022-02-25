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
logf()             log with EPS flooring
log10f()           log10 with EPS flooring
check_colab() 
```


+ **Defined Constants**
```
LOG10       log(10)
LOG2DB      10.0/LOG10
EPS_FLOAT   1.e-39  

```

+ **Kaldi Constants (default flooring)**
```
SIGEPS_FLOAT = 1.19209290e-7
SIGEPS_DB    = -69.23689          # scale*math.log(1.19209290e-7)  

```


### Signal Processing Routines

- spectrogram()       Fourier or mel spectrogram
- spg2mel()           converts Fourier Spectrogram to mel scale
- cepstrum()          waveform or spectrogram to cepstrum
- cep_lifter()        cepstrum inverse
- melcepstrum()

- mean_norm()           mean normalization, utterance based
- deltas()              delta computation / augmentation
- splice_frames()       splicing of frames
    
    

