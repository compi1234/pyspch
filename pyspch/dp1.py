# -*- coding: utf-8 -*-
"""
The modules in `dtw.py` contain basic implementations of Levenshtein and Weighted Edit Distance DP matching
The main purpose is for didactic demonstrations of  small systems 

Created on Jan 13, 2021
        
@author: compi
"""
import numpy as np
import pandas as pd

# tokenizer converts a text to tokens
def tokenizer(text,tolower=False):
    '''
    convert a text to a list of tokens
    '''
    if(tolower):
        text=text.lower()
    tokens = text.strip().split()
    return(tokens)

def print_align(align):
    '''
    prints an alignment given in DataFrame format
    '''
    with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
        print(align)

def print_edit_results(cts=None,align=None,trellis=None,Display=True):
    '''
    pretty prints results from edit_distance() 
    '''
    if df_align is not None:
        print(" == ALIGNMENT == ")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
            if(Display):
                display(align)
            else:
                print(align)
                
    if cts is not None:
        print("Results:\n#S=%d, #I=%d, #D=%d for %d tokens \nErr=%.2f%%" % cts )

    if trellis is not None:
        if(Display):
            display(trellis)
        
def alignment_to_counts(df):
    ''' 
    count nSUB/nINS/nDEL, nTOT and Err from an alignment dataframe
    Parameters:
    -----------
                df  type DataFrame, alignment as provided e.g. by wedit()
    Returns:
    --------
                (nsub,nins,ndel,ntot,err)   counts of SUB/INS/DEL and TOT and Err in %
        '''
    operands = df['O']
    Nsub = np.sum(operands=='S')
    Nins = np.sum(operands=='I')
    Ndel = np.sum(operands=='D')
    Nmatch = np.sum(operands=='M')
    Ntot = Nmatch + Ndel + Nsub 
    Err = (100.*(Nsub+Nins+Ndel)/Ntot)
    return(Nsub,  Nins, Ndel, Ntot, Err)
    
def lev_distance(seq1, seq2):
    '''
    Levenshtein Distance
        Finds the symmetric Levenshtein distance as the sum of INS/DEL/SUB 
        There is no backtracking, and no separate maintenance of INS/SUB/DEL 
        Each operation adds '1.' to the cost
    
    Parameters
    ----------
        seq1 : list
            tokens in list1 (either hypothesis or test)
        seq2 : list
            tokens in list2 (the other)
    
    Returns
   --------
        dist : int
            total number of edits
    '''
    Nx = len(seq1) 
    Ny = len(seq2) 
    prev = np.zeros(Ny+1)
    current = np.zeros(Ny+1)
    
    for j in range(0,Ny+1):
        current[j] = j

    for i in range(1, Nx+1):
        prev = current.copy()
        current[0] = prev[0]+1
        for j in range(1, Ny+1):
            if seq1[i-1] == seq2[j-1]:
                current[j] = min( prev[j]+1, prev[j-1], current[j-1]+1)
            else:    
                current[j] = min( prev[j], prev[j-1], current[j-1] ) + 1      
    return (current[Ny])


def edit_distance(x=[],y=[],wS=1.,wI=1.,wD=1.,Verbose=False):
    '''
    Weighted Edit Distance by DTW aligment allowing for SUB/INS/DEL

    Parameters
    ----------
    x : list (or str) 
        tokens in hypothesis/test
    y : list (or str)
        tokens in reference

    wS, wI, wD: float, defaults to 1.   
        edit costs for Substition, Insertion, Deletion

    Verbose : boolean, default=False
        if True highly Verbose printing of internal results (trellis, backtrace, .. )
    
    Returns
    -------
        dist : float
            weighted edit distance
        alignment : DataFrame
            alignment path as DataFrame with columns [ 'x', 'y', 'OPS' ]
        cts : list of int
            counts of [nsub,nins,ndel,Ny,nCx,nCy]
            if Compounds is False then last 2 arguments will be 0
        trellis :
            2D nd-array with trellis values
             
    '''
    if isinstance(x,str): x = list(x)
    if isinstance(y,str): y = list(y)        
    Nx = len(x) 
    Ny = len(y) 
    trellis = np.zeros((Nx+1, Ny+1))
    bptr = np.zeros((Nx+1, Ny+1, 2), dtype='int')
    edits = np.full((Nx+1, Ny+1),"Q",dtype='str')
     
    for i in range(1,Nx+1):
        trellis[i, 0] = i * wI
        bptr[i,0] = [i-1,0]
        edits[i,0] = 'I'
    for j in range(1,Ny+1):
        trellis[0, j] = j * wD
        bptr[0,j] = [0,j-1]
        edits[0,j] = 'D'

    # forward pass - trellis computation
    # indices i,j apply to the trellis and run from 1 .. N
    # indices ii,jj apply to the data sequence and run from 0 .. N-1
    for i in range(1, Nx+1):
        ii=i-1
        for j in range(1, Ny+1):
            jj=j-1

            # substitution or match
            score_SUB = trellis[i-1,j-1] + int(x[ii]!=y[jj]) * wS
            trellis[i,j] = score_SUB
            bptr[i,j] = [i-1,j-1]         
            if (x[ii]==y[jj]): edits[i,j] = 'M'            
            else: edits[i,j] = 'S'
                
            # insertion and deletions
            score_INS = trellis[i-1,j] + wI            
            if( score_INS < trellis[i,j] ):
                trellis[i,j] = score_INS
                bptr[i,j] = [i-1,j]
                edits[i,j] = 'I'
            score_DEL = trellis[i,j-1] + wD                
            if( score_DEL < trellis[i,j] ):
                trellis[i,j] = score_DEL
                bptr[i,j] = [i,j-1]
                edits[i,j] = 'D'      
                    
    # backtracking
    (ix,iy) = bptr[Nx,Ny]
    trace = [ (Nx-1,Ny-1) ]
    while( (ix>0) | (iy>0) ):
        trace.append( (ix-1,iy-1) )
        (ix,iy) = bptr[ix,iy]   
    trace.reverse()
    
    # recovering alignments as [ ( x_i, y_j, edit_ij ) ]
    # the dummy symbol '_' is inserted for counterparts of insertions, deletions
    #     and first element in compounds
    alignment = []
    for k in range(len(trace)):
        (ix,iy) = trace[k]
        ops = edits[ix+1,iy+1]
        if (ops == 'I') : 
            alignment.append( [ x[ix] , '_' , ops ] )
        elif (ops == 'D') : 
            alignment.append( [ '_' , y[iy] , ops ] ) 
        elif (ops == 'M') | (ops == 'S') : 
            alignment.append( [ x[ix] , y[iy] , ops ] )
            
    Nins = sum([ int(alignment[i][2]=='I') for i in range(len(alignment))])
    Ndel = sum([ int(alignment[i][2]=='D') for i in range(len(alignment))])
    Nsub = sum([ int(alignment[i][2]=='S') for i in range(len(alignment))])

    alignment_df = pd.DataFrame(alignment,columns=['x','y','O']).T
    trellis_df = pd.DataFrame(trellis.T, index=(['#']+y),columns=(['#']+x))
    
    dist = trellis[Nx,Ny]
    if (Verbose):
        print("Edit Distance: ", dist)
        print(trellis[0:,0:].T)
        print(edits[0:,0:].T)
        print(trace)
        print(alignment_df.transpose())
        print("Number of Words: ",Ny)
        print("Substitutions/Insertions/Deletions: ",Nsub,Nins,Ndel)


    Err = (100.*(Nsub+Nins+Ndel)/Ny)
    cts = (Nsub,  Nins, Ndel, Ny, Err)

    return(dist,alignment_df,cts,trellis_df)

