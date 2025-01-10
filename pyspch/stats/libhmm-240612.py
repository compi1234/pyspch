# Hidden Markov Models - libhmm
"""
The modules in `libhmm.py` contain a basic implementation of hidden Markov models
The main purpose is didactic demonstrations of (very) small systems 
with applications such as speech recognition in mind. 
Efficiency of the code is not a major concern

Author: Dirk Van Compernolle
Modification History:
12/12/2018 Created
11/02/2019 All list-like objects (states, labels, end_states, ... ) are now uniformly numpy arrays allowing for list indexing
14/11/2019: changed 'backtrace' to 'alignment'
19/11/2019: added routine backtrack() to class Trellis
22/11/2021: 
    The observation probabilities are now  computed by "an observation model" that is completely discoupled from the HMM 
    thus collapsing to a single HMM class and eliminating the different styles
    
    The observation model should support at least two method:
    obs_model.predict_log_prob(X):    computes log likelihoods for feature vector(s) X
    obs_model.predict_prob(X):        computes likelihoods for feature vector(s) X

11/01/2022:
    changed format of printing sequence probability to .2e
    changed buffer initialization in viterbi_recursion to -np.inf for log probs
    added .fillna("") to printing of  probs and obs_probs  dataframes

25/02/2022:
    adaptation to v0.6

11/05/2023:
    introduction of shared observations, leading to decoupling of state and observation indexing :
     - n_classes:  number of observation classes, if None inferred from obs_model  (skipped right now, problem ??)
     - obs_indx:   index mapper from state index to observation class index
    this also effected a small change in the initialization of state names unless specified 

26/03/2024:  bug fixes:
    - in trellis.plot_trellis(): no backpointer label should be generated if backpointer state index is "-1"
    - in hmm.viterbi_recursion():  bug fixing for low probs in log-style
    - excluding zero-prob related backpointers in first column of trellis  (not solidly implemented)
    - added .floor attribute in trellis() to implement the above
    - added logprob_floor attribute to class hmm

22/05/2024:
    - makes logprob_floor dominant over prob_floor
    - make probabilities by default float64
29/05/2024
    - added .draw() method for hmm
    
"""
import sys, os
from math import ceil, pow
import numpy as np
import pandas as pd
from IPython.display import display, HTML
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec  
import seaborn as sns
import copy
# import from graphviz if available; this only affects the module .draw()
try:
    from graphviz import Digraph
except:
    pass
    
#from pyspch.core.constants import EPS_FLOAT
from ..core.constants import EPS_FLOAT, EPS_LOG
from ..core import utils as u
from . import probdist as Densities

PROB_FLOOR = EPS_FLOAT
LOGPROB_FLOOR = EPS_LOG

# HMM Master Class
class HMM():
    """ 
    The HMM() class is the generic class for HMM models

    Attributes common to ALL instantiations of  HMM()
    ======================================================

    n_states  : int
        Number of states
    transmat  : array, shape (from_states, to_states)
        Transition Probability Matrix between states, indexed 
        (from_state,to_state)
    initmat :   array, shape (n_states, )
        Initial probability distribution over the states.
    states :    array, shape(n_states, )
        Names for states
    obs_indx : array, shape(n_states, )
        index mapping array that for each state specifies the class index to be used in the observation model
    obs_model:  a probdist(Densities) object that accepts all essential
        methods,  default(None) will select the dummy pass-through Probs()
        
    prob_style : string, "lin", "log" or "log10"  (some backwards compatibility for "prob" and "logprob")
        Do computations with probabilities or log-probabilities   
    prob_floor: probability flooring, set by default to PROB_FLOOR
        flooring is used when converting to logprobs
    
    Variables associated with an HMM 
    ================================
    
    X         : array, shape(n_samples, n_features)
        Feature Vector
        
    obs_probs : array, shape(n_samples, n_states)
        Observation Probabilities


    Methods 
    =======
    
    init_topology()
        initializes the state diagram
        
    set_probstyle()
        allows for switching between prob's and logprob's for computations
    
        
    print_model()
        pretty prints HMM transition and observation model parameters

    ==========
    """

    def __init__(self,n_states=1,transmat=None,initmat=None,states=None,obs_indx=None,end_states=None,obs_model=None,n_classes=None,prob_style="lin",logprob_floor=EPS_LOG,prob_floor=None):

        self._Debug = False
            
        self.prob_style = prob_style

        if logprob_floor is None:
            if prob_floor is not None:
                self.logprob_floor = np.log(prob_floor)
            else:
                self.logprob_floor = LOGPROB_FLOOR
        else:
            self.logprob_floor = logprob_floor
        self.prob_floor = np.exp(self.logprob_floor)
        
        # initialize the observation model
        if obs_model is None:
            self.obs_model = None # Densities.Prob()
        else:
            self.obs_model = obs_model
            # maybe the code below needs to be fixed; some models have classes_ others n_classes, others classes defined
            if hasattr(self.obs_model,"n_classes"): 
                self.n_classes = self.obs_model.n_classes
            elif hasattr(self.obs_model,"classes"): 
                self.n_classes = len(self.obs_model.classes)
            elif hasattr(self.obs_model,"classes_"): 
                self.n_classes = len(self.obs_model.classes_)
            else:
                self.n_classes = n_classes
            #self.n_classes = len(self.obs_model.classes_)
            
        # initialize  the state model
        # either the "states" array with state names should be given or the states will be
        # initialized as a range of integers 
        if (states is None):
            self.states = np.arange(n_states)
        else:
            self.states = np.array(states)      
        self.n_states = len(self.states)
        
        if obs_indx is None:
            self.obs_indx = np.arange(self.n_states)
        else: self.obs_indx = np.array(obs_indx)
        
        if (transmat is None):
            self.transmat = np.eye(self.n_states)
            if(self.prob_style == "log"):
                self.transmat = u.logf(self.transmat,logeps=self.logprob_floor)
        else:
            if(transmat.shape != (self.n_states,self.n_states)):
                sys.exit("ERROR(init_hmm): transition matrix of wrong size is given")
            self.transmat = transmat.astype('float64')

        if (initmat is None):
            self.initmat = np.zeros(self.n_states)
            self.initmat[0] = 1.0
            if(self.prob_style == "log"):
                self.initmat = u.logf(self.initmat,logeps=self.logprob_floor)
        else:
            if(initmat.size != self.n_states):
                sys.exit("ERROR(init_hmm): initial probability matrix of wrong size is given")
            self.initmat = initmat.astype('float64')  

        if end_states is None:
            self.end_states = np.arange(self.n_states)
        else:
            self.end_states = end_states

    def set_probstyle(self,prob_style):      
        self.transmat = u.convertf(self.transmat,iscale=self.prob_style,oscale=prob_style)
        self.initmat = u.convertf(self.initmat,iscale=self.prob_style,oscale=prob_style) 
        self.prob_style = prob_style
  
    def init_topology(self,type="lr",selfprob=0.5):
        if(type == "lr"):
            self.initmat = np.zeros(self.n_states)
            self.initmat[0] = 1.
            self.end_states = np.array([ self.n_states-1 ],'int')
            self.transmat = np.eye(self.n_states)
            for j in range(0,self.n_states-1):
                self.transmat[j,j]=selfprob    
                self.transmat[j,j+1]=1.0-selfprob
        elif(type == "ergodic"):
            self.initmat = np.ones(self.n_states)/self.n_states
            self.end_states = np.array([i for i in range(self.n_states)])
            self.transmat = (1-selfprob)/(self.n_states-1) * np.ones((self.n_states,self.n_states))
            for j in range(self.n_states):
                self.transmat[j,j]=selfprob          
        if(self.prob_style == "log"):
            self.initmat = np.log(u.floor(self.initmat,self.prob_floor))       
            self.transmat = np.log(u.floor(self.transmat,self.prob_floor)) 
            
    def print_model(self,what="all"):
        if (what == "all") | (what == "states"):
            print("\nHMM STATE MODEL\n=================\n")
            dfi = pd.DataFrame(self.initmat.reshape(1,-1),columns=self.states, index=["Pinit(.)"])
            display(dfi)
            states_str = self.states.astype('str')
            dft = pd.DataFrame(self.transmat.T,columns=states_str,
                index= ['P('+states_str[i]+'|.)' for i in range(0,self.n_states)])
            display(dft) 

        if (what == "all") | (what == "observations"):        
            print("\nOBSERVATION MODEL\n=================\n")
            try:
                self.obs_model.print_model(style=self.prob_style)
            except: 
                try:
                    self.obs_model.print_model()
                except:
                    print("Nothing to print\n")

    def draw(self,PROBS=False,PRINT_EPS=.00001,name='HMM'):
        ''' 
        .draw() returns a graph of the HMM model using the graphviz library
        INIT and END nodes are shown as double circles
        REGULAR nodes are shown as single circles
        with the PROBS flag set to TRUE transition probabilities are printed on the arcs
        PRINT_EPS puts a threshold on the (linear) probs to be shown

        '''
        g = Digraph(name,engine='dot',filename=name+'.gv')
        g.attr(size='20', splines="true", rankdir='LR', rank='same')
        g.attr('node', shape='circle', group='base', fontcolor='blue', fillcolor='blue', fontsize='12')
        g.attr('edge', arrowsize='0.5',fontsize='12')
        # A and I are linear versions of transmat and initmat independent of probstyle
        A = u.convertf(self.transmat,iscale=self.prob_style,oscale='lin')
        I = u.convertf(self.initmat,iscale=self.prob_style,oscale='lin')

        states = []
        (n,n) = A.shape
    
        for i in range(0,n):
            states.append("S"+str(i))
            if (i not in self.end_states) and ( I[i] <= PRINT_EPS) :
                    g.node(states[i],label=self.states[i]) 
    
        # create all begin and end nodes
        g.attr('node',shape='doublecircle')
        for i in range(0,n):
            if (i in self.end_states) or (I[i]> PRINT_EPS):
                g.node(states[i],label=self.states[i]) 
        
        for i in range(0,n):
            for j in range(0,n):
                if(A[i,j] > PRINT_EPS):
                    if PROBS:
                        g.edge(states[i],states[j],label=str(self.transmat[i,j]))
                    else:
                        g.edge(states[i],states[j])
        return(g)

        
    def observation_prob(self,X):
        """
        compute the observation probability for a feature vector
        X can be of size (n_samples,n_features) or (n_features,) or (n_samples,)
        
        The result will ALWAYS be (n_samples,n_classes) !!!
        """
        if self.obs_model is None: return(X)
        
        if X.ndim==1: # recast to 2d - best guess - if needed
            n = 1
            if hasattr(self.obs_model,'n_features_in_'):
                n=self.obs_model.n_features_in_
            elif hasattr(self.obs_model,'n_features_'):
                n=self.obs_model.n_features_
            elif hasattr(self.obs_model,'n_features'):
                n=self.obs_model.n_features  
            X = X.reshape(-1,n)

        if(self.prob_style == "log"):
            obs_prob = self.obs_model.predict_log_prob(X)
        else:
            obs_prob = self.obs_model.predict_prob(X)            
        return(obs_prob)
       
    def viterbi_recursion(self, X, prev_buffer):
        """ Viterbi Processing over one single frame
            internal loop for any Viterbi style decoder
        
        Paramters:
        ----------
        X :              shape(n_states,)     probabilities of single observation
        prev_buffer :    shape(n_states,)     current Viterbi buffer 
        
        Returns:
        --------
        buffer :         shape(n_states,)     updated Viterbi buffer
        backptr :        shape(n_states,)     backpointers for current frame
        """
        
        if(self.prob_style == "log"):
            buffer = np.full(prev_buffer.shape,self.logprob_floor,dtype='float64') 
            backptr = np.full(self.n_states,-1,dtype=int)
            for to_state in range(0,self.n_states):
                best_from_state = -1
                for from_state in range(0,self.n_states):
                    new = self.transmat[from_state,to_state] + prev_buffer[from_state]
                    if( new > buffer[to_state] ):
                        buffer[to_state] = new
                        backptr[to_state] = from_state
                buffer[to_state] = buffer[to_state] + X[to_state]
                # auto-pruning of unreachable states 
                if( buffer[to_state] <= self.logprob_floor ):
                    #buffer_to_state = self.logprob_floor
                    backptr[to_state] = -1

        else:
            buffer = np.zeros(prev_buffer.shape)
            backptr = np.full(self.n_states,-1,dtype=int)
            for to_state in range(0,self.n_states):
                for from_state in range(0,self.n_states):
                    new = self.transmat[from_state,to_state] * prev_buffer[from_state]
                    if( new > buffer[to_state] ):
                        buffer[to_state] = new
                        backptr[to_state] = from_state
                buffer[to_state] = buffer[to_state] * X[to_state]
                # auto-pruning of unreachable states 
                if( buffer[to_state] <= self.prob_floor ):
                    #buffer_to_state = self.prob_floor
                    backptr[to_state] = -1
                    
        return(buffer,backptr)


    def vitalign(self,X=None,end_states=None):
        '''
        Viterbi alignment for the current HMM model and observation sequence X
        This method allows to list allowed end_states, default is last state defined in the model

        Parameters:
        -----------

        
        Returns:
        --------
        Alignment        shape(n_observations,)     array of state indices
        Probability      float64                      Viterbi Probability
        '''
        trellis = Trellis(self)
        trellis.viterbi_pass(X)
        seq_prob,end_state = trellis._finalize(end_states=end_states)
        return  ( trellis.backtrace(end_state=end_state) , seq_prob )

#    def viterbi_train(self,X):
#        y = []
#        for XX in X:
#            y.append(self.align(X=XX))
#        self.fit(X,y)
        
    def fit(self,X,y,**kwargs):
        '''
        fit() takes a list of example observations X and a list of example alignments y
        '''
        # training of the observation probabilities
        self.obs_model.fit(np.concatenate(X),np.concatenate(y),**kwargs)
            
        # train initial and transition probabilities
        init_counts = np.zeros(self.n_states,dtype='float64')
        trans_counts = np.zeros((self.n_states,self.n_states),dtype='float64')         
        for (Xi,yi) in zip(X,y):
            init_counts[yi[0]] += 1.
            for j in range(1,Xi.shape[0]):
                trans_counts[ yi[j-1],yi[j] ] += 1.
        self.transmat = trans_counts / np.sum(trans_counts,axis=1)[:,np.newaxis]
        self.initmat = init_counts / np.sum(init_counts)
        if self.prob_style == "log":
            self.transmat = Spch.logf(self.transmat)
            self.initmat = Spch.logf(self.initmat)
   
    
# Defining a Trellis Class
class Trellis():
    """
    The Trellis calls is a generic container class for a trellis and computations related to the trellis

    All objects of the same size of trellis (probs, obs_probs, backptrs) 
    have the shape (n_samples,n_states), thus rows are added with each sample  (as in typical dataframes)
    To print/plot all these arrays are transposed
    
    Many of the attributes are dynamic (they grow as the trellis grows) and are optional (they are available 
    depending on the methods that have been run)
    Hence there can only be a limited amount of consistency checking and
    keeping everything consistent is left to the responsibility of the user.

    Attributes of a trellis are:
    ============================
        hmm             reference to hmm used to compute the trellis (required)
                        all attributes from the hmm are inherited by the trellis
        
        probs           trellis cell values   array, shape(n_samples,n_states), dtype='float64'
        obs_probs       obs. probs of frames  array, shape(n_samples,n_states), dtype='float64'
        backptrs        back pointers         array, shape(n_samples,n_states), dtype='int'
        alignment       Viterbi Alignment     array, shape(n_samples,), dtype='int'
        observations    processed observatons array, shape(n_samples,) or (n_samples,n_features)
        style           method applied        Viterbi (default), Forward (NIY)
        Normalize       column normalization  Boolean (default=False)
        floor           floor value under which everything should be interpreted as zero probability
        
        scale_vec       suggested scaling value per sample        float64(n_samples)
        end_state       best admissible end state
        seq_prob        sequence probability in end_state

        """
    
    def __init__(self,hmm,style='Viterbi',Normalize=False):
        """
        create and initialize trellis with reference to existing hmm
        process first observation

        """
        self.hmm = hmm
        self.style = style
        self.Normalize = Normalize
     
        self.n_samples = 0
        self.obs_probs  = np.ndarray((0,self.hmm.n_states),dtype='float64')
        self.probs  = np.ndarray((0,self.hmm.n_states),dtype='float64')
        self.backptrs = np.zeros(self.probs.shape,dtype='int') - 1
        
        # scaling factor to be applied in last column of trellis
        # create a trellis floor value that is somewhat higher than the hmm floor ... very heuristic and not waterproof !
        # 
        if self.hmm.prob_style == "lin": 
            self.scale  = 1.0
            self.floor = hmm.prob_floor 
        else: 
            self.scale = 0.0
            self.floor = hmm.logprob_floor
    
    def viterbi_pass(self,X):
        """
        X must be of dimension (s_samples,n_features)
        """
        for i in range(X.shape[0]):
            x = X[i:i+1,:]
            self._viterbi_step(x)
        self.seq_prob, self.end_state = self._finalize()
        
    def forward_pass(self,X):
        """
        this routine takes a SEQUENCE of OBSERVATIONS as argument 
        """
        for i in range(X.shape[0]):
            x = X[i:i+1,:]
            self._forward_step(x)
        self.seq_prob, self.end_state = self._finalize()
            
    def backtrace(self,end_state = None):
        if  self.style != 'Viterbi':
            print("Trellis.backtrace: Backtracing is only possible on Viterbi style Trellis")
            exit(-1)
        alignment = np.zeros((self.n_samples,),dtype='int')
        # be sure to finalize (might be duplicate)
        if end_state is None: end_state = self.end_state
        _, _  = self._finalize(end_states=[end_state])
        alignment[self.n_samples-1] = self.end_state
        
        for i in range(self.n_samples-1,0,-1):
            alignment[i-1]=self.backptrs[i,alignment[i]]
        return(alignment)
    
    def _col_norm(self,col):
        '''
        optional column normalization in a trellis
        with update of global scaling
        '''
        if self.Normalize:
            if self.hmm.prob_style == "lin":
                scale = np.max(col)
                col_norm = col/scale
                self.scale = self.scale * scale
            else:
                scale = np.max(col)
                col_norm = np.where(col>self.floor, col - scale, self.floor)
                self.scale = self.scale + scale
            return(col_norm) 
        else: return(col)
        
    def _observation_step(self,X):
        '''
        go to the observation model to compute b(X) of single frame
        returns:
        obs_probs[] as mapped onto the states by hmm.obs_indx 
        
        Remark: this is an efficient way of storing the obs_probs[] in the trellis,
        but it avoids explicit knowledge of the observation model and number of classes to the HMM model
        '''
        obs_probs = self.hmm.observation_prob(X.reshape(1,-1)).flatten() 
        return( obs_probs[self.hmm.obs_indx] )
    
    def _viterbi_step(self,X):
        """
        this routine takes EXACTLY ONE observation as argument
        and should have dimensions (n_features,)
        """

        obs_probs = self._observation_step(X)
        if self.n_samples == 0:
            t_, b_ = self._step0(obs_probs) 
        else:
            t_ , b_ = self.hmm.viterbi_recursion(obs_probs,self.probs[self.n_samples-1,:])
        

        self.obs_probs = np.r_[self.obs_probs,[obs_probs]]       
        self.probs = np.r_[self.probs,[self._col_norm(t_)]]
        self.backptrs = np.r_[self.backptrs,[b_]]
        self.n_samples += 1
        
    def _forward_step(self,X):
        """
        this routine takes EXACTLY ONE observation as argument
        """
        # reshaping is done as observation_prob expects (n_samples,n_features)
        obs_probs = self._observation_step(X)
        if self.n_samples == 0:
            t_,_ = self._step0(obs_probs) 
        else:
            t_ = np.zeros(self.hmm.n_states)
            prev_ = self.probs[self.n_samples-1,:]
            for to_state in range(0,self.hmm.n_states):
                for from_state in range(0,self.hmm.n_states):
                    t_[to_state] += self.hmm.transmat[from_state,to_state] * prev_[from_state]
                t_[to_state] = t_[to_state] * obs_probs[to_state]

        self.obs_probs = np.r_[self.obs_probs,[obs_probs]]      
        self.probs = np.r_[self.probs,[self._col_norm(t_)]]
        self.n_samples += 1    
        
    def _step0(self,obs_probs):
        """
        processes the first column of a trellis
        """
        if(self.hmm.prob_style == "log"):
            t_ = self.hmm.initmat + obs_probs
        else:
            t_ = self.hmm.initmat * obs_probs
        b_ = np.full(self.hmm.n_states,-1)
        
        # pruning of backpointers in first column if prob is too small (quite heuristic)
        for i in range(self.hmm.n_states):
            if self.hmm.initmat[i] > self.hmm.prob_floor: b_[i] = i
                
        return t_, b_       


    def _finalize(self,end_states = None):
        """
        find sequence probability and corresponding end_state, subjective to admissible end states 
        """
        # determine best admissible endstate
        endprobs = self.probs[self.n_samples-1,:]
        if end_states is None:
            end_states = self.hmm.end_states
            
        end_state = end_states[np.argmax( endprobs[end_states] )]
        seq_prob = endprobs[end_state] 

        # add accumulative scaling to end prob
        if self.hmm.prob_style == 'lin':
            seq_prob = seq_prob * self.scale
        else:
            seq_prob = seq_prob + self.scale
     
        return seq_prob,end_state


    
    def print_trellis(self,what='all',X=None,Titles=True):
        if what == 'all': 
            if self.style == 'Viterbi':
                what = ['obs_probs','probs','alignment','seq_prob']
            else:
                what = ['obs_probs','probs','seq_prob']

        if X is not None:
            if(Titles): print("\nObservations\n")
            if X.ndim ==1: 
                Xd = X.reshape(self.n_samples,1)
                indx = ['X']
            else: 
                Xd = X
                indx = [ 'x['+str(i)+']' for i in range(Xd.shape[1]) ]
            xdf = pd.DataFrame(Xd.T,index=['X'])
            display(xdf)
        else: xdf = None
        
        for w in what:
            if w == "obs_probs":
                fdf = pd.DataFrame(self.obs_probs.T,
                           columns=np.arange(self.n_samples),index = self.hmm.states ).fillna("")
                           #index = ['S'+str(i) for i in range(self.hmm.n_states)]) 
                             
                if(Titles): print("\nObservation Probabilities\n")
                display(fdf)

            elif w == "probs":
                pdf = pd.DataFrame(self.probs.T,
                           columns=np.arange(self.n_samples),index = self.hmm.states ).fillna("")
                           #index = ['S'+str(i) for i in range(self.hmm.n_states)])  
                if(Titles):
                    if self.style == "Viterbi": print("\nTrellis Probabilities (Viterbi)\n")
                    else: print("\nTrellis Probabilities (Forward)\n")
                display(pdf)

            elif w== "backpointers":
                bdf = pd.DataFrame(self.hmm.states[self.backptrs.T],
                           columns=np.arange(self.n_samples),index = self.hmm.states )
                           #index = ['S'+str(i) for i in range(self.hmm.n_states)])  
                if(Titles): print("\nBackpointers\n")
                display(bdf)
                
            elif w == "alignment":
                if(Titles): print("\nAlignment\n")
                alignment = self.backtrace().reshape(1,-1)
                adf = pd.DataFrame(self.hmm.states[alignment],index=['VIT-ALIGN'])
                display( adf )
                # display( pd.concat( (xdf,adf) ) )
                            
            elif w == "seq_prob":       
                print("\nSequence Probability: %.2e\n" % self.seq_prob)
            
            
    def plot_trellis(self,xticks=None,yticks=None,cmap=None,cmapf=None,
                     vmin=-10.,vmax=0.,fmt=".3f",fontsize=12,fontsize_backptrs=10,figsize=None,
                     plot_obs_probs=False,plot_norm=False,plot_values=True,plot_backptrs=False,plot_alignment=False):
        """
        plot_trellis(): trellis plot with multiple single axis plots
            observation probabilities are optionally added to the xticks
        """

        if xticks is None:
            xticks = np.array([str(i) for i in range(self.n_samples)])
        if yticks is None:
            yticks = self.hmm.states

        trellis = self.probs
        if plot_norm:
            trellis_n = copy.copy(trellis)
            if self.hmm.prob_style == "lin":
                fmax = np.amax(trellis,1)
                for j in range(self.n_samples):
                    trellis_n[j,:] = trellis[j,:]/fmax[j]
            else:
                fmax = np.amax(trellis,1)
                for j in range(self.n_samples):
                    trellis_n[j,:] = trellis[j,:]-fmax[j]            
        else:
            trellis_n = trellis

            
        fig = plt.figure(figsize=figsize)
        gs1 = gridspec.GridSpec(8, 1)
        gs1.update( hspace=0.15)
            
        if plot_obs_probs:
            if cmapf is None: cmapf = cmap
            axf = plt.subplot(gs1[0:3, 0])
            axt = plt.subplot(gs1[3:8, 0]) 
            sns.heatmap(self.obs_probs.T,ax=axf, vmin=vmin,vmax=vmax, 
                    xticklabels=xticks,yticklabels=yticks,
                    cmap=cmapf,square=False,cbar=False, linecolor='k',linewidth=0.2,
                    annot=plot_values,fmt=fmt,annot_kws={'fontsize':(fontsize-1),'color':'k'},
                       )
            axf.tick_params(axis='x',labelrotation=0.0,labeltop=True,labelbottom=False,bottom=False)
            axt.tick_params(axis='x',labelrotation=0.0,labeltop=False,labelbottom=False,bottom=False)
            axf.tick_params(axis='y',labelrotation=0.0,left=False)
        else:
            axt = plt.subplot(gs1[0:8, 0]) 
            axt.tick_params(axis='x',labelrotation=0.0,labeltop=True,labelbottom=False,bottom=False)
            axt.tick_params(axis='y',labelrotation=0.0,left=False)
            
        mask = self.probs < vmin
        if plot_values: annot = trellis.T
        else: annot=False
        sns.heatmap(trellis_n.T,ax=axt,vmin=vmin,vmax=vmax, mask=mask.T,
                    xticklabels=xticks,yticklabels=yticks,
                    cmap=cmap,square=False,cbar=False, linewidth=0.2, linecolor='k',
                    annot=annot,fmt=fmt,annot_kws={'fontsize':12,'color':'k'},
                    )
        axt.tick_params(axis='y',labelrotation=0.0,left=False)

        if(plot_backptrs):
            alignment = self.backtrace()
            for j in range(0,self.n_samples):
                for s in range(0,self.hmm.n_states):

                    if(not mask.T[s,j]):
                        if self.backptrs.T[s,j] == -1: bplabel = "" # No backpointer here !
                        else: bplabel = self.hmm.states[self.backptrs.T[s,j]]
                        if (alignment[j] == s) & plot_alignment:
                            axt.text(j+0.08,s+0.08,bplabel,ha="left",va="top",
                                fontweight='heavy',fontsize=fontsize_backptrs,color="k",rotation=-15,
                                     bbox={'boxstyle':'larrow,pad=.3', 'alpha':0.75, 'facecolor':'white'})                        
                        else:
                            axt.text(j+0.08,s+0.08,bplabel,ha="left",va="top",
                                fontweight='light',fontsize=fontsize_backptrs,color="k")

        plt.close()
        return(fig)