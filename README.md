# pyspch

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
