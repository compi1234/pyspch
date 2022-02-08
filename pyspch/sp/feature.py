import math
import numpy as np
import librosa
#from .constants import EPS_FLOAT

__all__=["cepstrum","mean_norm"]

# do preemhasis and padding
def fe_pad_and_preemp_(y,pad=None,preemp=None):
    '''
    constructs an array that is padded and preemphasized
    to make the output suitable for librosa processing with center = False (with a predictable number of frames)
        - for usage with librosa stft set pad to (n_fft-n_shift)/2
        - for time domain based feature extraction set pad to (n_length-n_shift)/2
    '''
    if pad is None: y_padded =y
    else: y_padded = np.concatenate((y[0:pad][::-1],y,y[:-pad-1:-1]))
    if preemp is None: 
        z = y_padded
    else:
        z=y_padded.copy()
        z[0]=(1.-preemp)*z[0]
        z[1:]= z[1:] - preemp*z[0:-1]
    return(z)

# returns the non truncated cepstrum if ncep is None, otherwise the truncated cepstrum
def cepstrum(spg,ncep=13):
    cep = fft.dct(spg,axis=0,type=3)
    if ncep is None:    return(cep)
    else: 
        return(cep[0:ncep,:])

def mean_norm(ftrs):
    ftrs_norm = (ftrs - ftrs.mean(axis=1,keepdims=True))
    return(ftrs_norm)

def melcepstrum(spg,ncep=13,nmel=80,sample_rate=8000):
    spgmel = Sps.spg2mel(spg,sample_rate=sample_rate,n_mels=nmel)
    cep = fft.dct(spgmel,axis=0,type=3,norm='ortho')
    if ncep is None:    return(cep)
    else: return(cep[0:ncep,:])
    
# returns evelope and residue spectra from cepstrum
# cep will first be padded to size 'nspec' before applying the dct
def cepstrum_inv(cep,ncep=13,nspec=128):
    if cep.shape[0] > cep.shape[0]:
        cep = np.vstack( (cep,np.zeros((nspec-cep.shape[0],cep.shape[1]))) )
    else:
        cep = cep[0:nspec,:]
    cep_t = np.zeros((nspec,cep.shape[1]))
    cep_t[0:ncep,:] = cep[0:ncep,:]
    #if cep.shape[0] > nspec : cep = cep[0:nspec,:]

    env = fft.idct(cep_t,axis=0,type=3)
    res = fft.idct(cep - cep_t,axis=0,type=3)
    return(env,res)

def feature_extraction(wavdata=None, spg=None,n_mels=80,sample_rate=8000,ncep=13,Deltas=True,Norm=None):
    #1. Spectral Estimation (Fourier Spectrogram)
    if spg is None:
        spg = Sps.spectrogram(wavdata,sample_rate=sample_rate,n_mels=None)
    
    #2. Mel scale transform
    if n_mels is not None:
        spg1 = Sps.spg2mel(spg,n_mels=n_mels,sample_rate=sample_rate)
    else: spg1 = spg
        
    #3. Cepstral transform
    if ncep is None: ftr1 = spg1
    else: ftr1 = cepstrum(spg1,ncep=ncep)
        
    #4. Add delta's
    if Deltas:
        deltas = librosa.feature.delta(ftr1,order=1)
        ftrs = np.vstack((ftr1,deltas))
    else: ftrs = ftr1
        
    #5. Mean and variance normalization
    if Norm is None:
        ftrs_n = ftrs
    elif Norm == "mean":
        ftrs_n = (ftrs - ftrs.mean(axis=1,keepdims=True) )
    elif Norm == "meanvar":
        ftrs_n = (ftrs - ftrs.mean(axis=1,keepdims=True) )/ ftrs.std(axis=1,keepdims=True)        

    return(ftrs_n)    