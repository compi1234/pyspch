import numpy as np
import pandas as pd
from IPython.display import display, Audio, HTML, clear_output
import ipywidgets as widgets
from ipywidgets import Box, HBox, VBox, Layout
import librosa


from .. import core as Spch     # .audio
from .. import sp as Sps      # spectrogram, spg2mel, cepstrum
from .display import SpchFig, PlotWaveform, PlotSpg, PlotSpgFtrs


import matplotlib.pyplot as plt



dw_5 = {'description_width': '50%'}
dw_4 = {'description_width': '40%'}
dw_3 = {'description_width': '30%'}
dw_2 = {'description_width': '20%'}
dw_1 = {'description_width': '10%'}
dw_0 = {'description_width': '0%'}

Symbols = { 'play':'\u25b6','reverse':'\u25C0' , 'pause':'\u23F8', 'stop': '\u23F9', 'record':'\u2b55'}

def box_layout(padding='1px',margin='0px',border='solid 1px black',**kwargs):
     return widgets.Layout(
#        display='flex', flex_flow='row', align_items='center',
        border= border,
        padding = padding,  # padding='2px 2px 2px 2px',  = white space inside the box; top, right, bottom, left
        margin=   margin,   # margin = '1px 1px 1px 1px', = white space around the box
         **kwargs
     )
    
def button_layout(padding='5px',margin='2px 5px 2px 5px',width='40px'):
    return widgets.Layout(
        border='solid 1px black',
        margin=margin,
        padding=padding,
        width = width,
        height = '40px',
        flex_shrink =2
     )


    
class MiniPlayer(widgets.HBox):
    '''
    MiniPlayer generates a simple widget with 2 buttons for Playing sound and Pausing it .  The widget is a handy replacement for the standard HTML5 audio control as it is smaller (and has fewer controls)
    
    usage: 
    > Player1 = MiniPlayer(data=wavdata,sample_rate=sr,border='')
    > display(Player1)
    
    IMPORTANT LIMITATION/BUG:
      Currently this only works if you stream your audio directly to your sound device.
      In Google Colab the playing is always handled by the HTML5 Audio element 
      and this will actually result in a double display of an audio widget
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
        

#########################################################
########  interactive Spectrogram
#########################################################
class iSpectrogram(VBox):
    '''
    iSpectrogram is an interactive spectrogram GUI where you can view:
        - as a basis: waveform and spectrogram
        - optional views of mel spectral / cepstral analysis
        - optional overlay with a segmentation
        - optionally a right hand pane with spectral slice views
        
    
    GUI controls include:
    - file selection
    - spectrogram parameters
    - postprocessing options
    - slider for frame selection

    input:
    ------
    type         int, type of spectrogram (default = 1)
                    1: Standard Spectrogram View with at least waveform and Fourier Spectrogram
                    2: Spectrogram View with a Right Hand Pane showing a spectral slice view of the 'current frame' as selected with a slider
    rhs_ratio    float, proportion of right hand side figure (slice plots); default=0.33
    aspect_ration float, aspect ratio of main figure; default = 0.4
    figwidth     float, figure width in inch (default=20.0) [best results if figwidth bigger than actual size] 
    dpi          int, mpl figure parameter (default=100)
    root         str, database name, default = 'https://homes.esat.kuleuven.be/~spchlab/data/'
    fname        str, filename, default = demo/friendly.wav'
    seg_pane     int (default=0): defines in which pane  the segmentation text is shown  (horizontal numbering top to bottom)
    RANGE_SLIDER boolean (default=False): adds an optional range selection slider at top of the GUI (reserved, not implemented)  
    DEBUG        boolean (default=False): if True run in Debug mode
    
    '''
    def __init__(self,dpi=100,figwidth=20.,rhs_ratio=0.33,aspect_ratio=0.4,type=1,seg_pane=0,
                root=None,
                fname='demo/friendly.wav',MELFB=False,CEP=False,RANGE_SLIDER=False,DEBUG=False):
        super().__init__()
        self.sample_rate = 1
        self.shift = 0.01
        self.length = 0.025
        self.preemp = 0.97
        self.nmels = 80
        self.melfb = MELFB
        self.ncep = 12
        self.cep = CEP
        self.wavdata = None
        self.root = root
        if self.root is None: self.root = "pkg_resources_data"
        self.fname = fname
        self.segfname = None
        self.seg_pane = seg_pane
        self.wavtimes = [0.,1.]
        self.seltimes = [0.,1.]
        self.frames = [0,1]
        self.frametime = 0.0
        self.frame = 0
        self.seg = None
        self.spg = None
        self.spgmel = None
        self.nparam = 0
        self.nfr = 0
        self.nshift = 0
        self.autoplay = False
        self.dpi = dpi
        self.figwidth = figwidth
        self.aspect = aspect_ratio
        self.rhs_ratio = rhs_ratio     # RHS vs. total display width
        self.type = type
        self.fig_range = None
        self.fig_main = None
        self.fig_rhs = None
        self.RANGE_SLIDER = False
        self.debug = DEBUG
        
        if self.type == 2: self.fig_ratio = 1.0 - self.rhs_ratio
        else: self.fig_ratio = 1.0
            
        # spectrogram controls
        self.wg_fshift = widgets.FloatSlider(value=1000*self.shift,min=5,max=50,step=5,description="Shift(msec)",style=dw_3)
        self.wg_flength = widgets.FloatSlider(value=1000*self.length,min=5,max=200,step=5,description="Length(msec)",style=dw_3)
        self.wg_preemp = widgets.FloatSlider(value=self.preemp,min=0.0,max=1.0,step=0.01,description="Preemphasis",style=dw_3)
        self.wg_melfb = widgets.Checkbox(value=self.melfb,description='Mel Filterbank',indent=True,style=dw_0)
        self.wg_nmels = widgets.IntSlider(value=self.nmels,min=10,max=128,step=1,description="#b",style=dw_3)
        self.wg_cep = widgets.Checkbox(value=self.cep,description='CEP/MFCC',indent=True,style=dw_0)
        self.wg_ncep = widgets.IntSlider(value=self.ncep,min=5,max=128,step=1,description="#c",style=dw_3)
        self.wg_melfb.layout.width='30%'
        self.wg_nmels.layout.width='70%'
        self.wg_cep.layout.width='30%'
        self.wg_ncep.layout.width='70%'
        self.wg_fshift.observe(self.fshift_observe,'value')
        self.wg_flength.observe(self.flength_observe,'value')
        self.wg_preemp.observe(self.preemp_observe,'value')        
        self.wg_nmels.observe(self.nmels_observe,'value')
        self.wg_melfb.observe(self.melfb_observe, 'value') 
        self.wg_ncep.observe(self.ncep_observe,'value')
        self.wg_cep.observe(self.cep_observe, 'value') 
        self.controls = VBox([ self.wg_fshift,self.wg_flength,self.wg_preemp, 
                               HBox([self.wg_melfb, self.wg_nmels]), 
                               HBox([self.wg_cep, self.wg_ncep]) ] ,
                               layout=box_layout() ) 
        
        # file controls
        self.audio_controls = widgets.Output()
        self.wg_root = widgets.Text(value=self.root,description="Root Dir: ",style=dw_2,continuous_update=False,layout=Layout(width='98%'))
        self.wg_root.observe(self.root_observe,'value') 
        self.wg_fname = widgets.Text(value=self.fname,description="WavFile ",style=dw_1,continuous_update=False,layout=Layout(width='98%'))
        self.wg_fname.observe(self.fname_observe,'value') 
        self.wg_segfname = widgets.Text(value=self.segfname,description="SegFile ",style=dw_1,continuous_update=False,layout=Layout(width='98%'))
        self.wg_segfname.observe(self.segfname_observe,'value')    
        # range slider
        self.wavrange = widgets.Output(layout=Layout(width='98%'))
        self.wg_range = widgets.FloatRangeSlider(value=[0.,1.],step=0.01,readout_format='.2f',
                            min=0.0,max=1.0,style=dw_1,
                            description='Range',continuous_update=False,readout=True,
                            layout=Layout(width='98%',padding='1px 2px 1px 3%') )   
        self.wg_range.observe(self.range_observe,'value')

        
        self.file_controls = VBox( [ self.wavrange,  self.wg_fname, self.wg_range, self.wg_segfname, self.audio_controls] ,   #wg_root used to be in here
                                  layout=box_layout(width='50%'))
        

        # frame slider
        slider_padding = '0px 2px 0px 5%' #   %.1f%%' % (20.-19.*self.fig_ratio) # this is an approximate hack
        self.frame_slider = widgets.IntSlider(value=0,step=1,
                            min=0,max=1,description='',continuous_update=False,readout=False)
        #self.frame_slider = widgets.FloatSlider(value=self.frame,step=self.shift,
        #                    min=0.0,max=1.0,
        #                   description='',continuous_update=False,readout=False)
        self.frame_slider.layout = Layout(width="99.5%",padding=slider_padding) 
        self.frame_slider.observe(self.frame_slider_observe,'value')

        #self.scr_range = VBox([ self.wavrange ],layout=box_layout())
        
        # Create all the Outputs: 
        self.out = widgets.Output(layout=Layout(padding="0px 5px 0px 0px")) 
        self.out2 = widgets.Output()
        self.logscr = widgets.Output()

        # putting it all together

        
        if self.type == 2:  # plot including SPECTRAL SLICES
            self.out.layout.height = '95%' 
            self.out2.layout.height = '97%' 
            self.controls.layout.width = '50%'
            self.file_controls.layout.width = '50%'        
            self.children = [
                    HBox([ 
                        VBox([self.out, self.frame_slider],layout=box_layout(  width=str(100.*self.fig_ratio)+"%"    ) ), 
                        VBox([self.out2], layout= box_layout(width=str(100.-100.*self.fig_ratio)+"%" )) ]
                        ) ,
                    HBox([   self.controls,  self.file_controls ]  ),
                    self.logscr 
                    ]
            if self.RANGE_SLIDER:  self.children = [self.scr_range] + self.children   # NEEDS WORK !!
        else: # plot without RHS
            self.controls.layout.width = '50%'
            self.file_controls.layout.width = '50%'
            self.out.layout = box_layout(width='100%')
            self.children = [ 
                # self.scr_range,
                self.out,
                HBox([   self.controls,  self.file_controls ]  ), 
                self.logscr ]                
            
        self.wav_update()
        self.update()
  
    def wav_update(self):
        with self.logscr:
            print("reading file: ",self.root + " + " + self.fname)
        self.wavdata, self.sample_rate = Spch.load_data(self.fname,root=self.root)  
        if self.wavdata is None:
            return
        # clear the segmentation
        self.seg = None
        self.segfname = ""
        #self.wg_segfname.value = ""  would trigger another update
        
        self.sample_period = 1./self.sample_rate

        # truncate waveform to a multiple of 10msec frames
        nshift_10 = round(0.01/self.sample_period)
        self.nfr = len(self.wavdata)//nshift_10
        self.nsamples = self.nfr * nshift_10
        self.wavdata = self.wavdata[0:self.nsamples]
        self.wavtimes = [0., len(self.wavdata)*self.sample_period]
        self.wg_range.min = self.wavtimes[0]
        self.wg_range.max = self.wavtimes[1]
        self.wg_range.value = self.wavtimes
        self.seltimes = self.wavtimes
        if self.debug:
            with self.logscr:
                print("setting seltimes:",self.seltimes)
                print("wg range max:",self.wg_range.max)
        self.frames = [0, int(self.wavtimes[1]/self.shift)]

        ## WARNING !!!
        ## The sequence of executing the following lines matters 
        ## as the observers() may come in between and behave unpredictably
        ## as you are forceably changing widget properties 
        ##
        self.frame_slider.max = self.frames[1]
        self.frame_slider.min = self.frames[0]
        self.frame_slider.value = 0 # (self.frames[1]+self.frames[0])//2
        #except:
        if self.debug:
            with self.logscr:
                #print("problem resetting slider, pls. try moving slider towards 0")
                print("current slider min-max-value:",self.frame_slider.min,self.frame_slider.max,self.frame_slider.value)
        self.frame = self.frame_slider.value                
        self.frametime = (self.frame+0.5)*self.shift
        
    def seg_update(self):
        # get segmentation
        # hack for timit segmentations  !!!! NOT ROBUST -- SHOULD BE CHANGED
        # dt = 1./self.sample_rate if self.segfname.split('/')[0]=='timit' else 1. 
        # improved in v0.8.2 with dt=None, dt will be chosen according to data type of segment boundaries (samples or times)
        #    hence only frame based segmentations can not be handled
        with self.logscr:
            print("reading file: ",self.root + " + " + self.segfname)
        self.seg = Spch.load_data(self.segfname,root=self.root,dt=None)
    
    def update(self,msg=None): 
        # uncomment the following lines to debug the observers
        if self.debug:
            with self.logscr:
                print("Updating : ",msg)
        if self.wavdata is None:
            with self.logscr:
                print("No wavdata found")
            return
        # round shift, length to sample
        self.length = round(float(self.sample_rate)*self.length)*self.sample_period
        if self.length < 0.002: self.length=0.002
        if self.shift > self.length: self.shift = self.length
        self.nshift = round(self.shift*self.sample_rate)      
        self.shift = self.nshift*self.sample_period
        
        # round number of frames and wavtimes to processed number of frames 
        self.nfr = (len(self.wavdata)//self.nshift) * self.nshift
        self.seltimes[1] = np.min([self.seltimes[1],self.wavtimes[1]])
        self.seltimes = [ round(float(self.sample_rate)*self.seltimes[0])/self.sample_rate,
                         round(float(self.sample_rate)*self.seltimes[1])/self.sample_rate ]

        self.frames = [ int(self.seltimes[0]/self.shift), int(self.seltimes[1]/self.shift) ] 
        self.frame_slider.max = self.frames[1]
        self.frame_slider.min = self.frames[0]
        self.frame = int(self.frametime/self.shift ) # np.max(0,int(self.frametime/self.shift - 0.5))
        self.frame_slider.value = self.frame
        if self.debug:
            with self.logscr:
                print("current slider min-max-value:",self.frame_slider.min,self.frame_slider.max,self.frame_slider.value)
        nextend = int((self.length-self.shift)*self.sample_rate/2.)
        self.frametimes = [self.frame*self.shift, (self.frame+1)*self.shift]
        self.framesamples = [self.frame*self.nshift, (self.frame+1)*self.nshift]
        self.winsamples = [self.framesamples[0]-nextend, self.framesamples[1]+nextend]
        self.wintimes = [self.winsamples[0]/self.sample_rate, self.winsamples[1]/self.sample_rate]
        
        # update range settings and figure
        self.fig_range = PlotWaveform(self.wavdata,sample_rate=self.sample_rate,ylabel=' ',xlabel=None,xticks=False,color='black',linewidth=1,
                            figsize=(10.,1.),dpi=self.dpi)
        self.fig_range.add_vrect(0.,self.seltimes[0],iax=0,color='#333',ec="#333",fill=True)
        self.fig_range.add_vrect(self.seltimes[1],self.wavtimes[1],iax=0,color='#333',ec="#333",fill=True)
        
        self.spg = Sps.spectrogram(self.wavdata,sample_rate=self.sample_rate,
                                  f_shift=self.shift,f_length=self.length,preemp=self.preemp,n_mels=None)
        self.spgmel = Sps.spg2mel(self.spg,sample_rate=self.sample_rate,n_mels=self.nmels)
        (self.nparam,self.nfr) = self.spg.shape
        img_ftrs = []
        img_labels = []
        segs = []
        # add melfilterbank view
        if self.melfb:
            img_ftrs += [self.spgmel]
            img_labels += ['mel '+str(self.nmels)]
            S = self.spgmel
            ceptype = 'cep'
        else:
            S = self.spg
            ceptype = 'cep'
        # add (mel) cepstral view
        if self.cep:
            ceps = Sps.cepstrum(S=S,n_cep=self.ncep)
            img_ftrs += [ ceps ]
            img_labels += [ceptype+str(ceps.shape[0])]
        # add segmentation
        try:
            seg1= Spch.read_seg_file(self.root+self.segfname)
            segs = [seg1] if seg1 is not None else []
        except:
            segs = []
   
        self.fig_main = PlotSpgFtrs(spgdata=self.spg,wavdata=self.wavdata,sample_rate=self.sample_rate,shift=self.shift,
                    dy=self.sample_rate/(2*(self.nparam-1)),img_ftrs=img_ftrs,img_labels=img_labels,  frames = self.frames,
                    figsize=(self.fig_ratio*self.figwidth,self.aspect*self.figwidth),dpi=self.dpi)
       
        for i in range(len(img_ftrs)+2):
            if i == self.seg_pane:
                self.fig_main.add_seg_plot(self.seg,iax=i,ypos=0.90,color="#444",size=14)
            else:
                self.fig_main.add_seg_plot(self.seg,iax=i,ypos=None,color="#222")

        with self.wavrange:
            clear_output(wait=True)
            display(self.fig_range) 
            
        if self.type == 2: # additional plotting when viewing the spectral slice
            self.fig_main.add_vrect(self.wintimes[0],self.wintimes[1],iax=0,color='#2F2')
            self.fig_main.add_vrect(self.frametimes[0],self.frametimes[1],iax=1,color='#222')
            for i in range(len(img_ftrs)):
                self.fig_main.add_vrect(self.frametimes[0],self.frametimes[1],iax=i+2,color='#222')
            self.plot_rhs(img_ftrs,img_labels)
            
        with self.out:
            clear_output(wait=True)
            display(self.fig_main)
            
        with self.out2:
            if self.type == 2:
                clear_output(wait=True)
                display(self.fig_rhs)
                
        with self.audio_controls:
            clear_output(wait=True)
            sample_range = [int(self.seltimes[0]*self.sample_rate),int(self.seltimes[1]*self.sample_rate)]
            display(Audio(data=self.wavdata[sample_range[0]:sample_range[1]],rate=self.sample_rate,autoplay=self.autoplay)) 
            # Miniplayer not working as desired on Colab
            #display( MiniPlayer(data=self.wavdata,sample_rate=self.sample_rate) )
            #display(Audio(data=self.wavdata,rate=self.sample_rate))

    def plot_rhs(self,ftrs,labels):           
        nftrs=0 if ftrs is None else len(ftrs)
        self.fig_rhs = SpchFig(row_heights=[1.,3.]+nftrs*[3.],figsize=((1.-self.fig_ratio)*self.figwidth,self.aspect*self.figwidth),dpi=self.dpi)
      
        sample_range = np.arange(self.winsamples[0],self.winsamples[1])
        self.fig_rhs.add_line_plot(self.wavdata[sample_range],iax=0,x=sample_range/self.sample_rate,color='#3F3',yrange=self.fig_main.axes[0].get_ylim())
        
        sample_range = np.arange(self.framesamples[0],self.framesamples[1])
        self.fig_rhs.axes[0].plot(sample_range/self.sample_rate,self.wavdata[self.framesamples[0]:self.framesamples[1]],color='#F00',linewidth=2)
        #self.fig_rhs.add_line_plot(self.wavdata[self.framesamples[0]:self.framesamples[1]],iax=0,x=sample_range/self.sample_rate,color='#F00',linewidth=2)  
        
        # spectral plot
        self.fig_rhs.add_line_plot(self.spg[:,self.frame],iax=1,xlabel='Freq (Hz) ',dx=self.sample_rate/(2.*(self.nparam-1)),linewidth=2)
        for i in range(nftrs):
            self.fig_rhs.add_line_plot(ftrs[i][:,self.frame],iax=i+2,xlabel=labels[i],linewidth=2)
        plt.close(self.fig_rhs)
        return()

    def root_observe(self,change):
        self.root=change.new
        #self.wav_update()
        #self.update()
        
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
        #self.frame_slider.step = self.shift
        self.update(msg="updating fshift")
        
    def flength_observe(self,change):
        self.length = change.new/1000.
        self.update(msg="updating flength")
        
    def preemp_observe(self,change):
        self.preemp = change.new
        self.update(msg="updating preemp")

    def nmels_observe(self,change):
        self.nmels = change.new
        self.update(msg="updating nmels")
    
    def melfb_observe(self,obj):
        self.melfb = obj.new
        self.update(msg="updating melfb")
        
    def ncep_observe(self,change):
        self.ncep = change.new
        self.update(msg="updating ncep")
        
    def cep_observe(self,change):
        self.cep = change.new
        self.update(msg="updating cep")   
        
    def range_observe(self,change):
        self.seltimes = list(change.new)
        if self.debug:
            with self.logscr:
                print(self.seltimes)
        self.update(msg="updating range")
        
    def frame_slider_observe(self,change):
        self.frame = change.new
        self.frametime = (self.frame+0.5)*self.shift
        self.update(msg="updating slider " + str(self.frametime))
        

class iRecorder(widgets.VBox):
    '''
    iRecorder is a GUI for speech recordings
    '''
    def __init__(self,sample_rate=16000,figsize=(12,3),dpi=72):
        super().__init__()
        
        sample_rates = [8000,11025,16000,22050,44100,48000]
        self.data = np.zeros(1024,dtype='float32')
        
        self.sample_rate = sample_rate
        self.rec_time = 3.0
        self.start_time = 0.0
        self.end_time = self.rec_time
        self.line_color = '#0000ff'
        self.figsize = figsize
        self.dpi = dpi
        self.filename = ""
        
        self.wg_play_button = widgets.Button(description=Symbols['play'],layout=button_layout())
        self.wg_record_button = widgets.Button(description=Symbols['record'],layout=button_layout())
        self.wg_pause_button = widgets.Button(description=Symbols['pause'],layout=button_layout())
        self.wg_clear_log_button = widgets.Button(description='Clear log')
        self.wg_save_button = widgets.Button(description='Save',layout=button_layout(width='120px'))
        self.wg_filename = widgets.Text(value=self.filename,description="File: ",
                                    style={'description_width': '15%'},continuous_update=False)

        self.wg_rectime = widgets.FloatSlider(   value=2.0, min=0.5, max= 10., step=0.5,
            readout_format="2.1f",
            description='Rec Time (sec):',style={'description_width': '30%'}, disabled=False)
        self.wg_samplerate = widgets.Dropdown(options=sample_rates,value=self.sample_rate,
                                              description="Sampling Rate",style={'description_width': '30%'})

        self.wg_start_time = widgets.FloatSlider(value=self.start_time, min=0., max= self.rec_time, 
            description='From (sec):',style={'description_width': '30%'}, disabled=False)
        
        self.wg_end_time = widgets.FloatSlider(value=self.rec_time, min=0., max= self.rec_time, 
            description='To (sec):',style={'description_width': '30%'}, disabled=False)        
        
        self.out = widgets.Output(layout=box_layout())
        self.record_box = widgets.HBox( 
                        [self.wg_play_button,self.wg_record_button,self.wg_pause_button,
                          self.wg_rectime,self.wg_samplerate],layout=box_layout(padding='10px'))
        self.save_box = widgets.HBox([
            self.wg_save_button, self.wg_filename,
            self.wg_start_time, self.wg_end_time],layout=box_layout(padding='10px'))
        self.logscr = widgets.Output()
        self.logscr_box = widgets.VBox([self.wg_clear_log_button,self.logscr],layout=box_layout(padding='10px'))
        self.UI = widgets.VBox([self.record_box,self.save_box],
                                layout=box_layout())

        # add as children
        self.children = [self.out, self.record_box,self.save_box, self.logscr_box] 
        
        self.wg_play_button.on_click(self.play_sound)       
        self.wg_record_button.on_click(self.record_sound)
        self.wg_pause_button.on_click(self.pause_sound)
        self.wg_clear_log_button.on_click(self.clear_log)
        self.wg_save_button.on_click(self.save_sound)
        self.wg_rectime.observe(self.rectime_observe,'value')
        self.wg_samplerate.observe(self.samplerate_observe,'value')
        self.wg_filename.observe(self.filename_observe,'value')
        self.wg_start_time.observe(self.start_time_observe,'value')
        self.wg_end_time.observe(self.end_time_observe,'value')
                
        self.plot_data()
        plt.close()


    def plot_data(self):
        with self.out:
            clear_output(wait=True)
            spg = Sps.spectrogram(self.data,sample_rate=self.sample_rate)
            self.fig = PlotSpg(spgdata=spg,wavdata=self.data,sample_rate=self.sample_rate,figsize=self.figsize,dpi=self.dpi)
            display(self.fig)
        
    def rectime_observe(self,change):
        self.rec_time = change.new
        self.wg_start_time.max = self.rec_time
        self.wg_end_time.max = self.rec_time
        
    def samplerate_observe(self,change):
        self.sample_rate = change.new

    def filename_observe(self,change):
        self.filename = change.new
        
    def start_time_observe(self,change):
        self.start_time = change.new
        
    def end_time_observe(self,change):
        self.end_time = change.new
        
    def pause_sound(self,b):  
        with self.logscr:
            Spch.audio.stop()
            
    def play_sound(self,b):
        Spch.audio.play(self.data,sample_rate=self.sample_rate,wait=False)

    def record_sound(self,b):      
        with self.logscr:
            clear_output()
            self.data = Spch.audio.record(self.rec_time,self.sample_rate,n_channels=1)
        self.plot_data()
        # self.play_sound(b)
        
    def save_sound(self,b):
        with self.logscr: 
            #print(self.start_time,self.end_time)
            i1 = int(self.start_time*self.sample_rate)
            i2 = int(self.end_time*self.sample_rate)
            if self.filename == "":
                print("You must specify a filename to save your data")
            else:
                print("saving data samples[%d:%d]  to %s"%(i1,i2,self.filename ) )
                Spch.audio.save(self.filename,self.data[i1:i2],self.sample_rate)
        
    def clear_log(self,b):
        with self.logscr: clear_output()
            
























################### OBSOLETE VERSION -- WILL BE DEPRECATED SOON #########################



class iSpectrogram1(Box):
    '''
         
    This is a legacy version of iSpectrogram() as it was prior to v0.8
    WILL BE DEPRECATED in the near future
    
    iSpectrogram1 is a spectrogram GUI where you can view:
        - as a basis: waveform and spectrogram
        - further mel-spectrum / cepstral analysis
        - optionally overlay with a segmentation

    
    GUI controls include:
    - file selection
    - spectrogram parameters
    - mel spectrum / cepstrum options
    - a double slider for selecting a part of the file frame
    

        - 'style': iSpectrogram can be run in 'horizontal' style (default) with the controls below the spectrogram view,
    or in 'vertical' style with the controls left of the spectrogram view
        - 
        
    input:
    ------
    A number of display options need to be specified at the time of calling the GUI
    
    style        str, must be 'horizontal'(default) or 'vertical'
    size         str, default '100%'
    figwidth     float, figure with in inch (default=12.0)
    dpi          int, mpl figure parameter (default=100)
    SEGTXT_WAV   boolean, (default=True) when True the text of a segmentation is visualized in the 'wav'-pane, otherwise in the spectrogram pane

    Defaults that can be changed in the GUI later
    root         str, database name, default = 'https://homes.esat.kuleuven.be/~spchlab/data/'
    fname        str, filename, default = demo/friendly.wav'
    MELFB        boolean, shows melfilterbank analysis (default=False)
    MFCC         boolean, shows (mel) cepstral analysis (default=False)
    
    
    '''
    
    def __init__(self,dpi=100,figwidth=12.,style='horizontal',size='100%',
                root='https://homes.esat.kuleuven.be/~spchlab/data/',
                fname='demo/friendly.wav',MELFB=False,MFCC=False,SEGTXT_WAV=True):
        super().__init__()
        self.sample_rate = 16000.
        self.shift = 0.01
        self.length = 0.025
        self.preemp = 0.97
        self.nmels = 80
        self.melfb = MELFB
        self.nmfcc = 12
        self.mfcc = MFCC
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
        self.nshift = None
        self.nparam = 0
        self.nfr = 0
        self.autoplay = False
        self.dpi = dpi
        self.figwidth = figwidth
        self.style = style
        self.layout.width = size
        
        self.fig_range = None
        self.fig_main = None
        self.SEGTXT_WAV = SEGTXT_WAV

        # spectrogram controls
        self.wg_fshift = widgets.FloatSlider(value=1000*self.shift,min=5,max=50,step=5,description="Shift(msec)",style=dw_3)
        self.wg_flength = widgets.FloatSlider(value=1000*self.length,min=5,max=200,step=5,description="Length(msec)",style=dw_3)
        self.wg_preemp = widgets.FloatSlider(value=self.preemp,min=0.0,max=1.0,step=0.01,description="Preemphasis",style=dw_3)
        self.wg_melfb = widgets.Checkbox(value=self.melfb,description='Mel Filterbank',indent=True,style=dw_0)
        self.wg_nmels = widgets.IntSlider(value=self.nmels,min=10,max=128,step=1,description="#b",style=dw_2)
        self.wg_mfcc = widgets.Checkbox(value=self.mfcc,description='Cepstra/MFCCs',indent=True,style=dw_0)
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
        self.file_controls = VBox( [  self.wg_root, self.wg_fname, self.wg_segfname , self.audio_controls] )
        
        # output widget for logging messages
        self.logscr = widgets.Output()
        
        # construct the main screen and range slider: 
        self.out = widgets.Output( layout=box_layout() ) 
        self.wavrange = widgets.Output(layout=Layout(width='99%'))
        self.wg_range = widgets.FloatRangeSlider(value=self.wavtimes,step=0.01,readout_format='.2f',
                            min=self.wavtimes[0],max=self.wavtimes[1],
                            description='',continuous_update=True,readout=False,
                            layout=Layout(width='98%',padding='1px 1px 1px 5%') )   
        self.wg_range.observe(self.range_observe,'value')
        self.wg_range.layout.width='99%'    
    
        # putting it all together
        self.layout.display = 'flex'
        self.layout.align_items = 'stretch'
        if self.style == 'horizontal':        
            self.layout.flex_flow = 'column'
            self.controls.layout = box_layout(width='35%')
            self.file_controls.layout =box_layout(width='35%')
            self.logscr.layout=box_layout(width='30%')
            #self.scr1 = VBox([ self.out, self.wavrange, self.wg_range ])
            self.scr2 = VBox([ self.wavrange, self.wg_range, 
                               HBox([ self.controls,  self.file_controls ,self.logscr ])], 
                              layout=box_layout())
            self.children =  [ self.out, self.scr2 ] 
        elif self.style == 'vertical':        
            #self.controls.layout = box_layout(width='100%')
            #self.file_controls.layout =box_layout(width='100%')
            #self.logscr.layout=box_layout(width='100%')
            self.scr1 = VBox([self.out, self.logscr] )
            self.scr2 = VBox([ self.controls,  self.file_controls ,
                              self.wg_range,self.wavrange] ,layout=box_layout(margin='2px 2px 2px 2px') )
            self.scr2.layout= box_layout(width='45%')
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
        
        self.spg = Sps.spectrogram(self.wavdata,sample_rate=self.sample_rate,
                                  f_shift=self.shift,f_length=self.length,preemp=self.preemp,n_mels=None)
        self.spgmel = Sps.spg2mel(self.spg,sample_rate=self.sample_rate,n_mels=self.nmels)
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
            #display(MiniPlayer(data=self.wavdata[sample_range[0]:sample_range[1]],sample_rate=self.sample_rate))
            display(Audio(data=self.wavdata[sample_range[0]:sample_range[1]],rate=self.sample_rate,autoplay=self.autoplay))           
            
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
            mfccs = Sps.cepstrum(S=S,n_cep=self.nmfcc)
            # mfccs = librosa.feature.mfcc(S=self.spgmel,sr=self.sample_rate,n_mfcc=self.nmfcc,dct_type=3) 
            img_ftrs += [ mfccs ]
            img_labels += [ceptype+str(mfccs.shape[0])]
   
        self.fig_main = PlotSpgFtrs(spgdata=self.spg,wavdata=self.wavdata,
                    sample_rate=self.sample_rate,shift=self.shift,
                    dy=self.sample_rate/(2*(self.nparam-1)),frames=self.frames,img_ftrs=img_ftrs,img_labels=img_labels,
                    figsize=(self.figwidth,0.5*self.figwidth),dpi=self.dpi)

        # plot the segmentation text either in the WAV or SPG pane
        if(self.SEGTXT_WAV):
            self.fig_main.add_seg_plot(self.seg,iax=0,ypos=0.85,color="#444",size=12)
            self.fig_main.add_seg_plot(self.seg,iax=1,ypos=None,color="#222")
        else:
            self.fig_main.add_seg_plot(self.seg,iax=0,ypos=None,color="#222",size=12)
            self.fig_main.add_seg_plot(self.seg,iax=1,ypos=0.90,color="#444")
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
        

            