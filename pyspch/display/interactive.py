import numpy as np
import pandas as pd
from IPython.display import display, Audio, HTML, clear_output
import ipywidgets as widgets
from ipywidgets import Box, HBox, VBox, Layout
import librosa


from .. import core as Spch     # .audio
from .. import sp       # spectrogram, spg2mel, cepstrum
from .display import SpchFig, PlotWaveform, PlotSpg, PlotSpgFtrs


import matplotlib.pyplot as plt



dw_5 = {'description_width': '50%'}
dw_4 = {'description_width': '40%'}
dw_3 = {'description_width': '30%'}
dw_2 = {'description_width': '20%'}
dw_0 = {'description_width': '0%'}


def box_layout(width='',height='',padding='1px',margin='1px',border='solid 1px black'):
     return Layout(
        border= border,
        padding = padding,  # padding='2px 2px 2px 2px',  = white space inside; top, right, bottom, left
        margin=   margin,   # margin = '1px 1px 1px 1px', = white space around the outside
        width = width,
        height = height
     )
    
Symbols = { 'play':'\u25b6','reverse':'\u25C0' , 'pause':'\u23F8', 'stop': '\u23F9', 'record':'\u2b55'}

def button_layout():
    return widgets.Layout(
        border='solid 1px black',
        margin='5px 5px 5px 5px',
        padding='5px 5px 5px 5px',
        width = '50px',
        height = '40px',
        flex_shrink =2
     )    

class MiniPlayer(widgets.HBox):
    '''
    MiniPlayer generates a simple widget with 2 buttons for Playing sound and Pausing it .  The widget is a handy replacement for the standard HTML5 audio control as it is smaller (and has fewer controls)
    
    usage: 
    > Player1 = MiniPlayer(data=wavdata,sample_rate=sr,border='')
    > display(Player1)
    
    '''
    def __init__(self,data,sample_rate=8000,width=None,border='',logscr=False):
        super().__init__()
        self.data = data
        self.sample_rate = sample_rate
        self.wg_play_button = widgets.Button(description=Symbols['play'],layout=button_layout())
        self.wg_pause_button = widgets.Button(description=Symbols['pause'],layout=button_layout())
        
        self.wg_play_box = widgets.VBox([self.wg_play_button])
        self.wg_pause_box = widgets.VBox([self.wg_pause_button])
           
        self.wg_play_button.on_click(self.play_sound)
        self.wg_pause_button.on_click(self.pause_sound) 
        if width is not None:
            if logscr: width = '500px'
            else: width = '125px'
        self.layout= box_layout(width=width,border=border)
        self.logscr = widgets.Output(layout=box_layout(border='1px solid blue',width='250px'))
        if logscr:
            self.children = [self.wg_play_box,self.wg_pause_box,self.logscr]
        else:
            self.children = [self.wg_play_box,self.wg_pause_box]              
        
    def play_sound(self,b):
        with self.logscr:
            print("Playing")
        Spch.audio.play(self.data,sample_rate=self.sample_rate,wait=False)
            
    def pause_sound(self,b):
        with self.logscr:
            print("Stop Playing")
        Spch.audio.stop()
        

class iSpectrogram(Box):
    '''
    iSpectrogram is an interactive spectrogram GUI where you can view:
        - as a basis: waveform and spectrogram
        - further spectral / cepstral analysis
        - optionally overlay with a segmentation
         
    
    User controls include:
    - file selection
    - spectrogram parameters
    - postprocessing options
    - a double slider for selecting a part of the file frame
    
    iSpectrogram can be run in 'horizontal' style (default) with the controls below the spectrogram view,
    or in 'vertical' style with the controls left of the spectrogram view
    
    input:
    ------
    style        str, must be 'horizontal'(default) or 'vertical'
    size         str, default '100%'
    figwidth     float, figure with in inch (default=12.0)
    dpi          int, mpl figure parameter (default=100)
    root         str, database name, default = 'https://homes.esat.kuleuven.be/~spchlab/data/'
    fname        str, filename, default = misc/friendly.wav'
    
    '''
    
    def __init__(self,dpi=100,figwidth=12.,style='horizontal',size='100%',
                root='https://homes.esat.kuleuven.be/~spchlab/data/',
                fname='misc/friendly.wav'):
        super().__init__()
        self.sample_rate = 16000.
        self.shift = 0.01
        self.length = 0.025
        self.preemp = 0.97
        self.nmels = 80
        self.melfb = False
        self.nmfcc = 12
        self.mfcc = False
        self.wavdata = None
        self.root = root
        self.fname = fname
        self.segfname = None
        self.wavtimes = [0., .88]
        self.seltimes = self.wavtimes 
        self.frames = [0, 1]
        self.spg = None
        self.spgmel = None
        self.seg = None
        self.nparam = 0
        self.nfr = 0
        self.autoplay = False
        self.dpi = dpi
        self.figwidth = figwidth
        self.style = style
        self.layout.width = size
        
        self.fig_range = None
        self.fig_main = None

        # spectrogram controls
        self.wg_fshift = widgets.FloatSlider(value=1000*self.shift,min=5,max=50,step=5,description="Shift(msec)",style=dw_3)
        self.wg_flength = widgets.FloatSlider(value=1000*self.length,min=5,max=200,step=5,description="Length(msec)",style=dw_3)
        self.wg_preemp = widgets.FloatSlider(value=self.preemp,min=0.0,max=1.0,step=0.01,description="Preemphasis",style=dw_3)
        self.wg_melfb = widgets.Checkbox(value=self.melfb,description='Mel Filterbank',indent=True,style=dw_0)
        self.wg_nmels = widgets.IntSlider(value=self.nmels,min=10,max=128,step=1,description="#b",style=dw_2)
        self.wg_mfcc = widgets.Checkbox(value=self.mfcc,description='Cepstra/MFFCs',indent=True,style=dw_0)
        self.wg_nmfcc = widgets.IntSlider(value=self.nmfcc,min=5,max=128,step=1,description="#c",style=dw_2)
        self.wg_melfb.layout.width='30%'
        self.wg_nmels.layout.width='70%'
        self.wg_mfcc.layout.width='30%'
        self.wg_nmfcc.layout.width='70%'
        self.wg_fshift.observe(self.fshift_observe,'value')
        self.wg_flength.observe(self.flength_observe,'value')
        self.wg_preemp.observe(self.preemp_observe,'value')        
        self.wg_nmels.observe(self.nmels_observe,'value')
        self.wg_melfb.observe(self.melfb_observe, 'value') 
        self.wg_nmfcc.observe(self.nmfcc_observe,'value')
        self.wg_mfcc.observe(self.mfcc_observe, 'value') 
        self.controls = VBox([ self.wg_fshift,self.wg_flength,self.wg_preemp, 
                               HBox([self.wg_melfb, self.wg_nmels]), 
                               HBox([self.wg_mfcc, self.wg_nmfcc]) ] 
                                ) 

        # file controls
        self.wg_root = widgets.Text(value=self.root,
                        description="Root Dir: ",style=dw_2,continuous_update=False,layout=Layout(width='98%'))
        self.wg_root.observe(self.root_observe,'value') 
        self.wg_fname = widgets.Text(value=self.fname,
                        description="Wav File: ",style=dw_2,continuous_update=False,layout=Layout(width='98%'))
        self.wg_fname.observe(self.fname_observe,'value') 
        self.wg_segfname = widgets.Text(value=self.segfname,
                        description="Seg File: ",style=dw_2,continuous_update=False,layout=Layout(width='98%'))
        self.wg_segfname.observe(self.segfname_observe,'value')         
        self.audio_controls = widgets.Output()
        self.file_controls = VBox( [  self.wg_root, self.wg_fname, self.wg_segfname ] )
        
        # output widget for logging messages
        self.logscr = widgets.Output()
        
        # construct the main screen and range slider: 
        self.out = widgets.Output( layout=box_layout() ) 
        self.wavrange = widgets.Output()
        self.wg_range = widgets.FloatRangeSlider(value=self.wavtimes,step=0.01,readout_format='.2f',
                            min=self.wavtimes[0],max=self.wavtimes[1],
                            description='',continuous_update=True,readout=False,
                            layout=Layout(width='98%',padding='0px 0% 0px 5%') )   
        self.wg_range.observe(self.range_observe,'value')
        self.wg_range.layout.width='99%'    
    
        # putting it all together
        self.layout.display = 'flex'
        self.layout.align_items = 'stretch'
        if self.style == 'horizontal':        
            self.layout.flex_flow = 'column'
            self.controls.layout = box_layout(width='35%')
            self.file_controls.layout =box_layout(width='35%')
            self.logscr.layout=box_layout(height='70%')
            self.scr1 = VBox([ self.out, self.wavrange, self.wg_range ])
            self.scr2 = HBox([ self.controls,  self.file_controls, 
                              VBox([ self.audio_controls,  self.logscr ], layout=box_layout(width='30% ')) ])
            self.children =  [ self.scr1, self.scr2 ] 
        elif self.style == 'vertical':        
            #self.controls.layout = box_layout(width='100%')
            #self.file_controls.layout =box_layout(width='100%')
            #self.logscr.layout=box_layout(width='100%')
            self.scr1 = VBox([self.out] )
            self.scr2 = VBox([ self.controls,  self.file_controls ,
                              VBox([self.wg_range,self.wavrange,self.audio_controls]), self.logscr ] )
            self.scr2.layout= box_layout(width='30%')
            self.children =  [ self.scr2, self.scr1 ]         
        self.wav_update()
        self.update()
  
    def wav_update(self):
        fname = self.root+self.fname
        #with self.logscr:
        #    print("audio file name",fname)
        self.wavdata, self.sample_rate = Spch.audio.load(fname)
        self.wavtimes = [0., len(self.wavdata)*(1./self.sample_rate)]
        self.wg_range.min = self.wavtimes[0]
        self.wg_range.max = self.wavtimes[1]
        self.wg_range.value = self.wavtimes
        self.seltimes = self.wavtimes
        self.nshift = int(self.shift*self.sample_rate)
        self.fig_range = PlotWaveform(self.wavdata,sample_rate=self.sample_rate,ylabel=None,xlabel=None,xticks=False,
                        figsize=(self.figwidth,0.1*self.figwidth),dpi=self.dpi)
        with self.wavrange:
            clear_output(wait=True)
            display(self.fig_range)        

    def seg_update(self):
        # get segmentation
        # hack for timit segmentations  !!!! NOT ROBUST -- SHOULD BE CHANGED
        dt = 1/self.sample_rate if self.segfname.split('/')[0]=='timit' else 1. 
        self.seg = Spch.read_seg_file(self.root+self.segfname,dt=dt)
        
    def update(self):     
        # round shift, length to sample
        self.shift = round(float(self.sample_rate)*self.shift)/self.sample_rate
        self.length = round(float(self.sample_rate)*self.length)/self.sample_rate
        self.seltimes = [ round(float(self.sample_rate)*self.seltimes[0])/self.sample_rate,
                         round(float(self.sample_rate)*self.seltimes[1])/self.sample_rate]
        if self.length < 0.002: self.length=0.002
        if self.shift > self.length: self.shift = self.length
        self.n_shift = int(self.shift*self.sample_rate)
        self.frames = [int(self.seltimes[0]/self.shift), int(self.seltimes[1]/self.shift)]
        
        self.fig_range = PlotWaveform(self.wavdata,sample_rate=self.sample_rate,ylabel=' ',xlabel=None,xticks=False,
                            figsize=(self.figwidth,1.0),dpi=self.dpi)
        self.fig_range.add_vrect(0.,self.seltimes[0],iax=0,color='#333',ec="#333",fill=True)
        self.fig_range.add_vrect(self.seltimes[1],self.wavtimes[1],iax=0,color='#333',ec="#333",fill=True)
        
        self.spg = sp.spectrogram(self.wavdata,sample_rate=self.sample_rate,
                                  f_shift=self.shift,f_length=self.length,preemp=self.preemp,n_mels=None)
        self.spgmel = sp.spg2mel(self.spg,sample_rate=self.sample_rate,n_mels=self.nmels)
        (self.nparam,self.nfr) = self.spg.shape
        #with self.logscr:
        #    print(self.shift,self.n_shift,self.spg.shape)

        self.plot1()
            
        with self.wavrange:
            clear_output(wait=True)
            display(self.fig_range) 
            
        with self.out:
            clear_output(wait=True)
            display(self.fig_main)
            
        with self.audio_controls:
            clear_output(wait=True)
            sample_range = [int(self.seltimes[0]*self.sample_rate),int(self.seltimes[1]*self.sample_rate)]
            display(MiniPlayer(data=self.wavdata[sample_range[0]:sample_range[1]],sample_rate=self.sample_rate))
            #display(Audio(data=self.wavdata[sample_range[0]:sample_range[1]],rate=self.sample_rate,autoplay=self.autoplay))           
            
    def plot1(self):
        img_ftrs = []
        img_labels = []
        
        # add melfilterbank view
        if self.melfb:
            img_ftrs += [self.spgmel]
            img_labels += ['mel '+str(self.nmels)]
            S = self.spgmel
            ceptype = 'mfcc'
        else:
            S = self.spg
            ceptype = 'cep'
        # add (mel) cepstral view
        if self.mfcc:
            mfccs = sp.cepstrum(S=S,n_cep=self.nmfcc)
            # mfccs = librosa.feature.mfcc(S=self.spgmel,sr=self.sample_rate,n_mfcc=self.nmfcc,dct_type=3) 
            img_ftrs += [ mfccs ]
            img_labels += [ceptype+str(mfccs.shape[0])]
   
        self.fig_main = PlotSpgFtrs(spgdata=self.spg,wavdata=self.wavdata,
                    sample_rate=self.sample_rate,shift=self.shift,
                    dy=self.sample_rate/(2*(self.nparam-1)),frames=self.frames,img_ftrs=img_ftrs,img_labels=img_labels,
                    figsize=(self.figwidth,0.5*self.figwidth),dpi=self.dpi)

        self.fig_main.add_seg_plot(self.seg,iax=0,ypos=0.85,color="#444",size=12)
        self.fig_main.add_seg_plot(self.seg,iax=1,ypos=None,color="#222")
        for i in range(len(img_ftrs)):
            self.fig_main.add_seg_plot(self.seg,iax=2+i,ypos=None,color="#222")

    #def plot2(self):
    #    self.fig = SpchFig()
            
    def root_observe(self,change):
        self.root=change.new
        
    def fname_observe(self,change):
        self.fname=change.new
        self.wav_update()
        self.update()
        
    def segfname_observe(self,change):
        self.segfname=change.new
        self.seg_update()
        self.update()
        
    def autoplay_observe(self,change):
        self.autoplay = change.new
        
    def fshift_observe(self,change):
        self.shift = change.new/1000.
        self.update()
        
    def flength_observe(self,change):
        self.length = change.new/1000.
        self.update()
        
    def preemp_observe(self,change):
        self.preemp = change.new
        self.update()

    def nmels_observe(self,change):
        self.nmels = change.new
        self.update()
    
    def melfb_observe(self,obj):
        self.melfb = obj.new
        self.update()
        
    def nmfcc_observe(self,change):
        self.nmfcc = change.new
        self.update()
    
    def mfcc_observe(self,obj):
        self.mfcc = obj.new
        self.update()
       
    def range_observe(self,change):
        self.seltimes = change.new
        self.update()
        

#########################################################
########  Spg2
#########################################################
class iSpectrogram2(VBox):
    '''
    iSpectrogram2 is an interactive spectrogram GUI where you can view:
        - as a basis: waveform and spectrogram
        - further spectral / cepstral analysis
        - optionally overlay with a segmentation
        
    iSpectrogram2 has in the left hand pane a classical spectrogram view and in
    the right hand pane the view of single frame/slice.  
    
    User controls include:
    - file selection
    - spectrogram parameters
    - postprocessing options
    - slider for frame selection

    input:
    ------
    size         str, default '100%'
    figwidth     float, figure with in inch (default=12.0)
    dpi          int, mpl figure parameter (default=100)
    root         str, database name, default = 'https://homes.esat.kuleuven.be/~spchlab/data/'
    fname        str, filename, default = misc/friendly.wav'
        
    '''
    def __init__(self,dpi=100,figwidth=12.,size='100%',
                root='https://homes.esat.kuleuven.be/~spchlab/data/',
                fname='misc/friendly.wav'):
        super().__init__()
        self.sample_rate = 1
        self.shift = 0.01
        self.length = 0.025
        self.preemp = 0.97
        self.nmels = 80
        self.melfb = False
        self.nmfcc = 12
        self.mfcc = False
        self.wavdata = None
        self.root = root
        self.fname = fname
        self.segfname = None
        self.wavtimes = [0., 1.]
        self.seltimes = self.wavtimes 
        self.frames = [0, 1]
        self.frame = 0
        self.seg = None
        self.spg = None
        self.spgmel = None
        self.nparam = 0
        self.nfr = 0
        self.autoplay = False
        self.dpi = dpi
        self.figwidth = figwidth
        #self.layout.width = size
        self.fig_ratio = .66     # LHS display vs RHS
        self.fig_range = None
        self.fig_main = None
        self.fig_rhs = None

        # spectrogram controls
        #self.wg_fshift = widgets.FloatSlider(value=self.shift,min=0.005,max=0.050,step=0.005,description="Shift(msec)",readout_format='.3f',style=dw_3)
        #self.wg_flength = widgets.FloatSlider(value=self.length,min=0.005,max=0.200,step=0.005,description="Length(msec)",readout_format='.3f',style=dw_3)
        self.wg_fshift = widgets.FloatSlider(value=1000*self.shift,min=5,max=50,step=5,description="Shift(msec)",style=dw_3)
        self.wg_flength = widgets.FloatSlider(value=1000*self.length,min=5,max=200,step=5,description="Length(msec)",style=dw_3)
        self.wg_preemp = widgets.FloatSlider(value=self.preemp,min=0.0,max=1.0,step=0.01,description="Preemphasis",style=dw_3)
        self.wg_melfb = widgets.Checkbox(value=self.melfb,description='Mel Filterbank',indent=True,style=dw_0)
        self.wg_nmels = widgets.IntSlider(value=self.nmels,min=10,max=128,step=1,description="#b",style=dw_3)
        self.wg_mfcc = widgets.Checkbox(value=self.mfcc,description='Cepstra/MFFCs',indent=True,style=dw_0)
        self.wg_nmfcc = widgets.IntSlider(value=self.nmfcc,min=5,max=128,step=1,description="#c",style=dw_3)
        self.wg_melfb.layout.width='30%'
        self.wg_nmels.layout.width='70%'
        self.wg_mfcc.layout.width='30%'
        self.wg_nmfcc.layout.width='70%'
        self.wg_fshift.observe(self.fshift_observe,'value')
        self.wg_flength.observe(self.flength_observe,'value')
        self.wg_preemp.observe(self.preemp_observe,'value')        
        self.wg_nmels.observe(self.nmels_observe,'value')
        self.wg_melfb.observe(self.melfb_observe, 'value') 
        self.wg_nmfcc.observe(self.nmfcc_observe,'value')
        self.wg_mfcc.observe(self.mfcc_observe, 'value') 
        self.controls = VBox([ self.wg_fshift,self.wg_flength,self.wg_preemp, 
                               HBox([self.wg_melfb, self.wg_nmels]), 
                               HBox([self.wg_mfcc, self.wg_nmfcc]) ] ,
                               layout=box_layout(width='50%') ) 
        #self.controls.width = '50%'
        
        # file controls
        self.audio_controls = widgets.Output()
        self.wg_root = widgets.Text(value=self.root,description="Root Dir: ",style=dw_2,continuous_update=False,layout=Layout(width='98%'))
        self.wg_root.observe(self.root_observe,'value') 
        self.wg_fname = widgets.Text(value=self.fname,description="Wav File: ",style=dw_2,continuous_update=False,layout=Layout(width='98%'))
        self.wg_fname.observe(self.fname_observe,'value') 
        self.wg_segfname = widgets.Text(value=self.segfname,description="Seg File: ",style=dw_2,continuous_update=False,layout=Layout(width='98%'))
        self.wg_segfname.observe(self.segfname_observe,'value')         
        self.file_controls = VBox( [  self.wg_root, self.wg_fname, self.wg_segfname, self.audio_controls] ,
                                  layout=box_layout(width='50%'))
        
        # slider and audio controls 
        self.wg_range = widgets.FloatSlider(value=self.frame,step=self.shift,
                            min=self.wavtimes[0],max=self.wavtimes[1],
                            description='',continuous_update=True,readout=False,
                            layout = box_layout(width=str(100.*self.fig_ratio)+"%" ,padding='0px 2px 0px 4%') )

        self.wg_range.observe(self.range_observe,'value')
        self.slider_controls = self.wg_range
                
        # Main and log Outputs: 
        self.out = widgets.Output( layout=Layout(width=str(100.*self.fig_ratio)+"%") )
        self.out2 = widgets.Output( layout=Layout(width=str(100.-100.*self.fig_ratio)+"%") )
        self.logscr = widgets.Output()

        # putting it all together
        self.children =  [ VBox([ HBox([  self.out, self.out2 ] ), self.slider_controls ], layout=box_layout()),
                          HBox([   self.controls,  self.file_controls ]  ), 
                           self.logscr ]
        
        self.wav_update()
        self.update()
  
    def wav_update(self):
        self.wavdata, self.sample_rate = Spch.audio.load(self.root+self.fname)  
        self.wavtimes = [0., len(self.wavdata)*(1./self.sample_rate)]
        self.frames = [0, int(self.wavtimes[1]/self.shift)]
        self.wg_range.min = self.wavtimes[0]
        self.wg_range.max = self.wavtimes[1]
        
        self.wg_range.value = (self.wavtimes[1]+self.wavtimes[0])/2
        self.frame = int(self.wg_range.value/self.shift)
        self.seltimes = self.wavtimes

        self.fig_range = PlotWaveform(self.wavdata,sample_rate=self.sample_rate,ylabel=None,xlabel=None,xticks=False,
                        figsize=(self.figwidth,0.1*self.figwidth),dpi=self.dpi)       
    def seg_update(self):
        # get segmentation
        # hack for timit segmentations  !!!! NOT ROBUST -- SHOULD BE CHANGED
        dt = 1/self.sample_rate if self.segfname.split('/')[0]=='timit' else 1. 
        self.seg = Spch.read_seg_file(self.root+self.segfname,dt=dt)
        
    def update(self):     
        # round shift, length to sample
        self.shift = round(float(self.sample_rate)*self.shift)/self.sample_rate
        self.length = round(float(self.sample_rate)*self.length)/self.sample_rate
        self.seltimes = [ round(float(self.sample_rate)*self.seltimes[0])/self.sample_rate,
                         round(float(self.sample_rate)*self.seltimes[1])/self.sample_rate]
        if self.length < 0.002: self.length=0.002
        if self.shift > self.length: self.shift = self.length
        self.nshift = int(self.shift*self.sample_rate)
        
        self.frame = int(self.seltime/self.shift)
        nextend = int((self.length-self.shift)*self.sample_rate/2)

        self.seltimes = [self.frame*self.shift, (self.frame+1)*self.shift]
        self.selsamples = [self.frame*self.nshift, (self.frame+1)*self.nshift]
        self.winsamples = [self.selsamples[0]-nextend, self.selsamples[1]+nextend]
        self.wintimes = [self.winsamples[0]/self.sample_rate, self.winsamples[1]/self.sample_rate]
        
        self.spg = sp.spectrogram(self.wavdata,sample_rate=self.sample_rate,
                                  f_shift=self.shift,f_length=self.length,preemp=self.preemp,n_mels=None)
        self.spgmel = sp.spg2mel(self.spg,sample_rate=self.sample_rate,n_mels=self.nmels)
        (self.nparam,self.nfr) = self.spg.shape
        img_ftrs = []
        img_labels = []
        segs = []
        # add melfilterbank view
        if self.melfb:
            img_ftrs += [self.spgmel]
            img_labels += ['mel '+str(self.nmels)]
            S = self.spgmel
            ceptype = 'mfcc'
        else:
            S = self.spg
            ceptype = 'cep'
        # add (mel) cepstral view
        if self.mfcc:
            mfccs = sp.cepstrum(S=S,n_cep=self.nmfcc)
            # mfccs = librosa.feature.mfcc(S=self.spgmel,sr=self.sample_rate,n_mfcc=self.nmfcc,dct_type=3) 
            img_ftrs += [ mfccs ]
            img_labels += [ceptype+str(mfccs.shape[0])]
        # add segmentation
        try:
            seg1= Spch.read_seg_file(self.root+self.segfname)
            segs = [seg1] if seg1 is not None else []
        except:
            segs = []
   
        self.fig_main = PlotSpgFtrs(spgdata=self.spg,wavdata=self.wavdata,sample_rate=self.sample_rate,shift=self.shift,
                    dy=self.sample_rate/(2*(self.nparam-1)),img_ftrs=img_ftrs,img_labels=img_labels,
                    figsize=(self.fig_ratio*self.figwidth,0.5*self.figwidth),dpi=self.dpi)
        self.fig_main.add_seg_plot(self.seg,iax=0,ypos=0.85,color="#444",size=12)
        self.fig_main.add_seg_plot(self.seg,iax=1,ypos=None,color="#222")
        
        #self.fig_main.add_vrect(self.seltimes[0],self.seltimes[1],iax=0,color='#F22')
        self.fig_main.add_vrect(self.wintimes[0],self.wintimes[1],iax=0,color='#2F2')
        self.fig_main.add_vrect(self.seltimes[0],self.seltimes[1],iax=1,color='#222')
        for i in range(len(img_ftrs)):
            self.fig_main.add_vrect(self.seltimes[0],self.seltimes[1],iax=i+2,color='#222')
            
        self.plot_rhs(img_ftrs,img_labels)
            
        with self.out:
            clear_output(wait=True)

            display(self.fig_main)
            
        with self.out2:
            clear_output(wait=True)
            display(self.fig_rhs)
            
        with self.audio_controls:
            clear_output(wait=True)
            #sample_range = [int(self.seltimes[0]*self.sample_rate),int(self.seltimes[1]*self.sample_rate)]
            display( MiniPlayer(data=self.wavdata,sample_rate=self.sample_rate) )
            #display(Audio(data=self.wavdata,rate=self.sample_rate,autoplay=self.autoplay))

    def plot_rhs(self,ftrs,labels):
        nftrs=0 if ftrs is None else len(ftrs)
        self.fig_rhs = SpchFig(row_heights=[1.,3.]+nftrs*[3.],figsize=((1.-self.fig_ratio)*self.figwidth,0.5*self.figwidth),dpi=self.dpi)
        sample_range = np.arange(self.winsamples[0],self.winsamples[1])
        self.fig_rhs.add_line_plot(self.wavdata[sample_range],iax=0,x=sample_range/self.sample_rate,color='#3F3',yrange=self.fig_main.axes[0].get_ylim())
        sample_range = np.arange(self.selsamples[0],self.selsamples[1])
        self.fig_rhs.axes[0].plot(sample_range/self.sample_rate,self.wavdata[self.selsamples[0]:self.selsamples[1]],color='#F00')
        self.fig_rhs.add_line_plot(self.spg[:,self.frame],iax=1,xlabel='Freq (Hz) ',dx=self.sample_rate/(2.*(self.nparam-1)))
        for i in range(nftrs):
            self.fig_rhs.add_line_plot(ftrs[i][:,self.frame],iax=i+2,xlabel=labels[i])
        plt.close(self.fig_rhs)
        return()

    def root_observe(self,change):
        self.root=change.new
        
    def fname_observe(self,change):
        self.fname=change.new
        self.wav_update()
        self.update()
        
    def segfname_observe(self,change):
        self.segfname=change.new
        self.seg_update()
        self.update()
        
    def autoplay_observe(self,change):
        self.autoplay = change.new
        
    def fshift_observe(self,change):
        self.shift = change.new/1000.
        self.wg_range.step = change.new/1000.
        self.update()
        
    def flength_observe(self,change):
        self.length = change.new/1000.
        self.update()
        
    def preemp_observe(self,change):
        self.preemp = change.new
        self.update()

    def nmels_observe(self,change):
        self.nmels = change.new
        self.update()
    
    def melfb_observe(self,obj):
        self.melfb = obj.new
        self.update()
        
    def nmfcc_observe(self,change):
        self.nmfcc = change.new
        self.update()
    
    def mfcc_observe(self,obj):
        self.mfcc = obj.new
        self.update()
       
    def range_observe(self,change):
        self.seltime = self.wg_range.value
        self.update()