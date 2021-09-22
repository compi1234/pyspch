# -*- coding: utf-8 -*-
# libhmm_plot.py   a collection of printing and plotting routines for HMM's
# 
"""
Created on Mon Feb 25 09:44:40 2019

@author: compi
@created: 25/02/2019

This library contains a variety of pretty print / plot routines for simple HMM models
WARNING: many of these routines are ad hoc and should be considered as an example rather
    than as a fool proof generic routine; i.e. for some models the output will look fine
    for others it may be dismal
    
14/11/2019: changed 'backtrace' to 'alignment'
"""

# do all the imports
import sys, os
import numpy as np
import pandas as pd
from IPython.display import display, HTML
import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns

import pyspch.libhmm as libhmm


# ok if using hmmplot by itself, but probably not if mixed with other modules
#mpl.rcParams['figure.figsize'] = [8.0, 8.0]
#mpl.rcParams['ps.papersize'] = 'A4'
#mpl.rcParams['xtick.labelsize'] = 14
#mpl.rcParams['ytick.labelsize'] = 14
#mpl.rcParams['axes.titlepad'] = 15
#mpl.rcParams['axes.titlesize'] = 'large'
#mpl.rcParams['axes.linewidth'] = 2
#mpl.rc('lines', linewidth=3, color='k')
ldesign = 25
cmap = sns.light_palette("caramel",ldesign,input="xkcd")
my_cmap = cmap[1:20] 



def plot_model(self,prob_style=[],cmap=[],figsize=(14,4)):
        
    fig, ax = plt.subplots(1,2,figsize=figsize,sharey=True,constrained_layout=True)
    if prob_style == "prob":
        vmn = 1.e-5
        vmx = 1.0
    else:
        vmn = -25.0
        vmx = 0.0
        
    xx= np.zeros((self.n_states,self.n_states+1),dtype=float)
    xx[:,0] = self.startprob
    xx[:,1:] = self.transmat
    mask = xx < vmn

    xticks = ['INIT']+['to-'+s for s in self.states]    
    sns.heatmap(xx,vmin =vmn, vmax=vmx,  mask=mask,
                ax=ax[0], xticklabels=xticks, yticklabels=self.states,fmt=".3f",
                square=True, cbar=False, cmap=cmap, linewidth=1, linecolor='k',
                annot=True,annot_kws={'fontsize':12,'color':'k'}          )
    ax[0].set_title("Init & Transition Probabilities",fontdict={'fontsize':15},pad=40)
       
    mask = self.emissionprob < vmn
    sns.heatmap(self.emissionprob,vmin =vmn, vmax=vmx,  mask=mask,
                ax=ax[1], xticklabels=self.labels,yticklabels=self.states,fmt=".3f",
                square=True, cbar=False, cmap=cmap, linewidth=1, linecolor='k',
                annot=True,annot_kws={'fontsize':12,'color':'k'}           )
    ax[1].set_title("Observation Probabilities",fontdict={'fontsize':15},pad=40)
    
    for axi in ax:
        axi.tick_params(axis='x',labelrotation=0.0,labeltop=True,labelbottom=False,bottom=False)
        axi.tick_params(axis='y',labelrotation=0.0,left=False)
        
    plt.close()
    return(fig)
    

def plot_trellis(self,X=None,plot_frameprobs=False,xticks=[],yticks=[],cmap=[],vmin=-10.,vmax=0.,fmt=".3f",figsize=(15,5),
                 plot_norm=False,plot_values=True,plot_backptrs=False,plot_alignment=False,fontsize_backptrs=10):
    """
    plot_trellis(): trellis plot with multiple single axis plots
        state_likelihoods are optionally added to the xticks
    """
    if len(X) == 0 :
        print("ERROR(plot_trellis): X is not specified\n")
        return
    n_frames = len(X)
    
    frameprobs, trellis, backptrs, alignment = self.viterbi_trellis(X)
    
    if not xticks:
        try:
            xticks = [ self.labels[x]  for x in X ]
        except:
            xticks = np.array([str(i) for i in range(0,n_frames)])
    if not yticks:
        yticks = self.states

    if plot_norm:
        trellis_a = trellis.copy()
        if self.prob_style == "lin":
            fmax = np.amax(trellis,1)
            for j in range(0,len(X)):
                trellis[j,:] = trellis[j,:]/fmax[j]
        else:
            fmax = np.amax(trellis,1)
            for j in range(0,len(X)):
                trellis[j,:] = trellis[j,:]-fmax[j]            
    else:
        trellis_a = trellis
        
    if(len(xticks) >100 ) : # old code   
        f,axf = plt.subplots(figsize=figsize)
        sns.heatmap(frameprobs.T,ax=axf,vmin=vmin,vmax=vmax, 
                    xticklabels=xticks,yticklabels=yticks,
                    cmap=cmap,square=True,cbar=False, linewidth=0, linecolor='k',
                    annot=True,fmt=fmt,annot_kws={'fontsize':12,'color':'k'},
                    )
        axf.tick_params(axis='x',labelrotation=0.0,labeltop=True,labelbottom=False,bottom=False)
        axf.tick_params(axis='y',labelrotation=0.0,left=False)
        plt.show()
        xticks = []
    elif plot_frameprobs:
        Nl,Ns = frameprobs.shape
        for j in range(Nl):
            for i in range(Ns):
                xticks[j] += "\n"+ str(np.round(frameprobs[j,i],3))
            xticks[j] += "\n"
        
    fig,axt = plt.subplots(figsize=figsize)
    mask = trellis < vmin
    sns.heatmap(trellis.T,ax=axt,vmin=vmin,vmax=vmax, mask=mask.T,
                xticklabels=xticks,yticklabels=yticks,
                cmap=cmap,square=True,cbar=False, linewidth=1.2, linecolor='k',
                annot=trellis_a.T,fmt=fmt,annot_kws={'fontsize':12,'color':'k'},
                )
    axt.tick_params(axis='x',labelrotation=0.0,labeltop=True,labelbottom=False,bottom=False)
    axt.tick_params(axis='y',labelrotation=0.0,left=False)

    if(plot_backptrs):
        for j in range(0,len(X)):
            for s in range(0,self.n_states):

                if(not mask.T[s,j]):
                    bplabel = self.states[backptrs.T[s,j]]
                    if (alignment[j] == s) & plot_alignment:
                        axt.text(j+0.08,s+0.08,bplabel,ha="left",va="top",
                            fontweight='heavy',fontsize=fontsize_backptrs,color="k",rotation=-15,
                                 bbox={'boxstyle':'larrow,pad=.3', 'alpha':0.75, 'facecolor':'white'})                        
                    else:
                        axt.text(j+0.08,s+0.08,bplabel,ha="left",va="top",
                            fontweight='light',fontsize=fontsize_backptrs,color="k")

# now add the backtrace as second x-axis labels at the bottom
    #if(plot_alignment):
    #    for j in range(0,len(X)):
    #        axt.text(j+0.5,self.n_states+0.05,self.states[alignment[j]],ha="center",va="top",
    #                        fontweight='heavy',fontsize=12,color="b")

    #plt.show()
    plt.close()
    return(fig)
 
    
#################################
def plot_trellis2(self,X=None,xticks=[],yticks=[],cmap=[],cmapf=[],vmin=-10.,vmax=0.,
                 plot_backptrs=False,plot_alignment=False,plot_frameprobs=False,plot_norm=False,
                 fontsize=12,fmt=".3f",figsize=(15,5)):
    """
    plot_trellis2(): trellis plotting using subplots
    """
    import matplotlib.gridspec as gridspec    
    
    if len(X) == 0 :
        print("ERROR(plot_trellis): X is not specified\n")
        return
    n_frames = len(X)    
    frameprobs, trellis, backptrs, alignment = self.viterbi_trellis(X)
    if plot_norm:
        trellis_p = trellis.copy()
        if self.prob_style == "lin":
            fmax = np.amax(trellis_p,1)
            for j in range(0,len(X)):
                trellis_p[j,:] = trellis_p[j,:]/fmax[j]
        else:
            fmax = np.amax(trellis_p,1)
            for j in range(0,len(X)):
                trellis_p[j,:] = trellis_p[j,:]-fmax[j]            
    else:
        trellis_p = trellis
        
    if not xticks:
        try:
            xticks = self.labels[X]
        except:
            xticks = np.array([str(i) for i in range(0,n_frames)])
    if not yticks:
        yticks = self.states

    fig = plt.figure(figsize=figsize)
    gs1 = gridspec.GridSpec(6, 1)
    gs1.update( hspace=0.05)
   
    if plot_frameprobs:
        axf = plt.subplot(gs1[0:2, 0])
        axt = plt.subplot(gs1[2:6, 0]) 
        sns.heatmap(frameprobs.T,ax=axf,vmin=vmin,vmax=vmax, 
                    xticklabels=xticks,yticklabels=[],
                    cmap=cmapf,square=True,cbar=False, linecolor='k',linewidth=0.0,
                    annot=True,fmt=fmt,annot_kws={'fontsize':(fontsize-1),'color':'k'},
                    )
        axf.tick_params(axis='x',labelrotation=0.0,labeltop=True,labelbottom=False,bottom=False)
        axt.tick_params(axis='x',labelrotation=0.0,labeltop=False,labelbottom=False,bottom=False)
    else:
        axt = plt.subplot(gs1[0:6, 0])
        axt.tick_params(axis='x',labelrotation=0.0,labeltop=True,labelbottom=False,bottom=False)
        
    mask = trellis < vmin
    
    sns.heatmap(trellis_p.T,ax=axt,vmin=vmin,vmax=vmax, mask=mask.T,
                xticklabels=xticks,yticklabels=yticks,
                cmap=cmap,square=True,cbar=False, linewidth=1.2, linecolor='k',
                annot=trellis.T,fmt=fmt,annot_kws={'fontsize':fontsize,'color':'k'},
                )
    if plot_frameprobs:
        axt.tick_params(axis='x',labelrotation=0.0,labeltop=False,labelbottom=False,bottom=False)
    else:
        axt.tick_params(axis='x',labelrotation=0.0,labeltop=True,labelbottom=False,bottom=False)        
    axt.tick_params(axis='y',labelrotation=0.0,left=False)
    

    for j in range(0,len(X)):
        for s in range(0,self.n_states):
            if(not mask.T[s,j]):
                bplabel = "("+self.states[backptrs.T[s,j]]+")"
                if(plot_alignment and (s == alignment[j]) ):
#                    axt.text(j+0.03,s+0.1,'*',ha="left",va="top",
                    axt.text(j+0.95,s+0.5,'*',ha="right",va="center",
                        fontweight='heavy',fontsize=fontsize+2,color="blue")
                if(plot_backptrs):
                    axt.text(j+0.03,s+0.5,bplabel,ha="left",va="center",
                        fontweight='light',fontsize=fontsize,color="k")
    
    # plt.show()    
    plt.close()
    return(fig)