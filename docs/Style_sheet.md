# Programming style and conventions
#
Speech Processing Parameters
'''
wavdata :       numpy float 32, sampled data; size (n_samples,) or (n_channels,n_samples)
                equivalence: y, waveform
n_samples :     int, number of samples
n_channels :    int, number of channels

channels :      array of int's , selected channels
sample_rate :   int, sampling frequency (default=16000)
seconds :       float , time in seconds to record or save (default=2.0)
io_device :     string, 'sd' for sounddevice, 'js' for javascript

n_frames :      int, number of frames
preemp :        float, preemphasis (default=0.95)
f_length :      frame length in seconds (default=0.03)
f_shift :       frame shift in seconds (default=0.01)
n_length :      frame length in samples
n_shift :       frame shift in samples
n_fft :         int, number of fft coefficients (default=512)
n_mels :         int, number of mel channels
window :        string, window type (default='hamm')
n_param :       int, number of parameters in a frame
frames :        (int,int), frame range  ( should be slice ? )
S :             numpy float 32, power spectogram; size (n_param,n_frames)
spgdata :       numpy float 32, spectrogram like data; size (n_param, n_frames)
mode :          string, the representation of the spectrogram, one of \['power','magnitude','dB'\]

dx :            float, sample spacing on x-axis
dy :            float, sample spacing on y-axis
dt :            float, sample period in seconds
x0 :            float, starting position on x-axis
y0 :            float, starting position on y-axis
xrange :        (float,float) , x-axis range. None stands for reuse, 'tight' stands for tight x-axis
yrange :        (float,float) , y-axis range. None stands reuse, 'tight' stands for 20% headroom on y-axis
xax :           float array, values for x-axis (overriding dx)
yax :           float array, values for y-axis (overriding dy)

xpos :          text positioning in relative units of axis width (0. .. 1.)
ypos :          text positioning in relative units of axis height (0. .. 1.)

Lines :         boolean, to draw vertical segmentation lines
segdf :         pandas DataFrame, segmentation with fields \['t0,'t1','seg'\]
segwav :        segmentation to be added to a waveform axis
segspg :        segmentation to be added to a spectrogram axis
title :         string, Figure Title
xlabel :        string, label for x-axis
ylabel :        string, label for y-axis
txt :           string, some text
kwargs :        dict, arguments to be passed to subfunctions
txtargs :       dict, plot arguments for text  (typically passed to ax.text())
lineargs :      dict, plot arguments for lines (typically passed to ax.vlines())


fig :           Figure object
iax :           axis index (for row grids only) or indices (for row+column grids), numbering from 0
row_heights :   float array, relative height ratios for a row grid, defaults to [1]
col_widths :    float array, relative width ratios for columns, defaults to [1]