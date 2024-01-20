import numpy as np
import scipy.sparse 
import pylab as plt
import LZ76
import pci_LZ76

def binJproc(J,N0,Nbegin=0,Nboot=500,alpha=0.01,boot='max'):
    '''
        J       evoked currents (nTime x nArea x nTrials)
        N0      number of time points of the prestimulus phase 
        Nbegin  how much to discard from baseline (0 = nothing is removed)
        Nboot   number of bootstraps
        alpha   the alpha level for acceptance of significant activities 
        boot    type of synthetic boostraped descriptor (max is the default)
    '''
    nTime,nArea,nTrials=J.shape
    base = np.mean(J[:N0,:,:],axis=(0,2))                                
    sub = np.kron(base.reshape((nArea,1)),np.ones((1,nTime-Nbegin)))    # baseline to subtract
    norm = np.std(J[:N0,:,:],axis=(0,2),ddof=1)                              
    Jmean = np.mean(J,axis=2)
    Jmean = Jmean.swapaxes(0,1) - sub

    NUM = np.kron(np.ones((1,N0)),base.reshape((nArea,1)))
    DEN = np.kron(np.ones((1,N0)),norm.reshape((nArea,1)))

    # bootstraps 
    randontrialsT=np.random.randint(0,nTrials,nTrials)
    Bootstraps=np.zeros((Nboot,N0))     
    for per in range(Nboot):
        ET=np.zeros((nArea,N0))              
        for j in range(nTrials): 
            randonsampT=np.random.randint(0,N0,N0)    
            ET+=J[randonsampT,:,randontrialsT[j]].swapaxes(0,1)
        ET=(ET/nTrials-NUM)/DEN # computes a Z-value 
        if boot=='max': Bootstraps[per,:] = np.max(np.abs(ET),axis=0)   # maximum statistics in space
        if boot=='mean': Bootstraps[per,:] = np.mean(np.abs(ET),axis=0) # mean statistics in space (ref. D'Andola et al. )

    # computes threshold for binarization depending on alpha value 
    Bootstraps=np.sort(np.reshape(Bootstraps,(Nboot*N0)))
    calpha=1-alpha 
    calpha_index=int(np.round(calpha*Nboot*N0))-1               
    TT=norm*Bootstraps[calpha_index]                                    # computes threshold based on alpha 
    Threshold=np.kron(np.ones((1,nTime-Nbegin)),TT.reshape((nArea,1)))  # set the same threshold for each time point 
    binJ=np.array(np.abs(Jmean)>Threshold,dtype=bool) 
    return binJ
    
def normalization(D):
    '''
        correction for low p1,p0 ?!
    '''
    L = D.shape[0] * D.shape[1]
    p1 = sum(1.0 * (D.flatten() == 1)) / L
    p0 = 1 - p1
    if p1*p0:
        H = -p1 * np.log2(p1) -p0 * np.log2(p0)
    else:
        print('p0=%g\np1=%g\n'%(p0,p1))
        H=0.

    S = (L * H) / np.log2(L)

    return H, S

def sort_binJ(binJ):
    ''' sort binJ as in Casali et al 2013
    '''
    SumCh=np.sum(binJ,axis=1)
    Irank=SumCh.argsort()[::-1]
    return binJ[Irank,:]

BaseSort=2.
def sort_binJ(binJ,whatsort='STD'):
    '''
    '''
    _,nTIME=binJ.shape

    if whatsort=='STD':
        SumCh=np.sum(binJ,axis=1)
        Irank=SumCh.argsort()[::-1]
        binJsort=binJ[Irank,:]

    if (whatsort=='EG1') or (whatsort=='EG'): # the standard EG=EG1 
        binCODE=BaseSort**np.arange(nTIME) 
        SumCh=np.sum(binJ,axis=1)
        Irank=SumCh.argsort()[::-1]
        binJsortTMP=binJ[Irank,:]  
        # sort the blocks 
        SumCh=np.sum(binJsortTMP,axis=1)
        Su=np.unique(SumCh)    
        binJ2c=np.copy(binJsortTMP)
        for S in Su:
            I=np.where(SumCh==S)[0]
            y=np.sum(binJsortTMP[I,:]*binCODE,axis=1)
            J=np.argsort(y)[::-1]
            binJsortTMP[I,:]=binJ2c[I[J],:]
        binJsort=binJsortTMP

    if whatsort=='EG2': # in analogy to S2 
        binCODE=BaseSort**(nTIME-np.arange(nTIME))
        SumCh=np.sum(binJ,axis=1)
        Irank=SumCh.argsort()[::-1]
        binJsortTMP=binJ[Irank,:]
        # sort the blocks
        SumCh=np.sum(binJsortTMP,axis=1)
        Su=np.unique(SumCh)    
        binJ2c=np.copy(binJsortTMP)
        for S in Su:
            I=np.where(SumCh==S)[0]
            y=np.sum(binJsortTMP[I,:]*binCODE,axis=1)
            J=np.argsort(y)[::-1]
            binJsortTMP[I,:]=binJ2c[I[J],:]
        binJsort=binJsortTMP

    if whatsort=='S1':
        binCODE=BaseSort**np.arange(nTIME) 
        y=np.sum(binJ*binCODE,axis=1)  
        J=np.argsort(y)[::-1]
        binJsort=binJ[J,:]
                    
    if whatsort=='S2':
        binCODE=BaseSort**(nTIME-np.arange(nTIME))
        y=np.sum(binJ*binCODE,axis=1)  
        J=np.argsort(y)[::-1]
        binJsort=binJ[J,:]

    return binJsort

def computePCI(binJ,t,tlim=[],nrun=10):
    '''
    '''
    binJcp=binJ.copy()
    if type(binJcp)==scipy.sparse.coo.coo_matrix: binJcp=binJcp.toarray()
    binJcp=sort_binJ(binJcp)
    if len(tlim)==2:
        idx=np.where((t>tlim[0])&(t<tlim[1]))[0]
        binJcp=binJcp[:,idx]
    Hsrc,norm_factor=normalization(binJcp)
    complexity=LZ76.lz_complexity_2D(binJcp) 
    norm_factor=LZ76.pci_norm_factor(binJcp) 
    return Hsrc,complexity /norm_factor

def load_binJ(fn,nArea=76):
    '''
    '''
    d=fn
    nTime=len(d['time'])
    binJ=np.zeros((nArea,nTime),dtype=bool)
    binJ[d['r'],d['c']]=1
    return binJ,d['time']

def plot_binJ(binJ,time,lab='significant activities', color='k', ms=4, plot_sum=True, subplots=(1,1,1)):
    '''
    '''
    r,c=np.where(binJ)
    plt.subplot(subplots[0],subplots[1],subplots[2])
    plt.plot(time[c],r,'.', c=color, ms=ms)
    if plot_sum:
        sum_binJ=binJ.sum(axis=0)
        plt.plot(time,sum_binJ,'r-',lw=2)
    # decorate plot 
    plt.xlabel('time (ms)',fontsize=8)
    plt.ylabel(lab,fontsize=8)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)    
    plt.xlim(time[0],time[-1])
    plt.ylim(-.25,binJ.shape[0]+0.25)   
    #plt.tight_layout()
    
# Lempel-Ziv complexity
from numpy import *
from numpy.linalg import *
from scipy import signal
from scipy.signal import hilbert
from scipy.stats import ranksums
from scipy.io import savemat
from scipy.io import loadmat
from random import *
from itertools import combinations
from pylab import *

'''
Python code to compute complexity measures LZc, ACE and SCE as described in "Complexity of multi-dimensional spontaneous EEG decreases during propofol induced general anaesthesia"

Author: m.schartner@sussex.ac.uk
Date: 09.12.14

To compute the complexity meaures LZc, ACE, SCE for continuous multidimensional time series X, where rows are time series (minimum 2), and columns are observations, type the following in ipython:

execfile('CompMeasures.py')
LZc(X)
ACE(X)
SCE(X)


Some functions are shared between the measures.
'''

def Pre(X):
    '''
        Detrend and normalize input data, X a multidimensional time series
    '''
    ro,co=shape(X)
    Z=zeros((ro,co))
    for i in range(ro):
        Z[i,:]=signal.detrend(X[i,:]-mean(X[i,:]), axis=0)
    return Z


##########
'''
LZc - Lempel-Ziv Complexity, column-by-column concatenation
'''
##########

def cpr(string):
    '''
    Lempel-Ziv-Welch compression of binary input string, e.g. string='0010101'. It outputs the size of the dictionary of binary words.
    '''
    d={}
    w = ''
    i=1
    for c in string:
        wc = w + c
        if wc in d:
            w = wc
        else:
            d[wc]=wc
            w = c
            i+=1
    return len(d)

def str_col(X):
    '''
     Input: Continuous multidimensional time series
     Output: One string being the binarized input matrix concatenated comlumn-by-column
     '''
    ro,co=shape(X)
    TH=zeros(ro)
    M=zeros((ro,co))
    for i in range(ro):
        M[i,:]=abs(hilbert(X[i,:]))
        TH[i]=mean(M[i,:])

    s=''
    for j in range(co):
        for i in range(ro):
            if M[i,j]>TH[i]:
                s+='1'
            else:
                s+='0'

    return s

def LZc(X):
    '''
    Compute LZc and use shuffled result as normalization
    '''
    X=Pre(X)
    SC=str_col(X)
    M=list(SC)
    shuffle(M)
    w=''
    for i in range(len(M)):
        w+=M[i]
    return cpr(SC)/float(cpr(w))

##########
'''
ACE - Amplitude Coalition Entropy
'''
##########

def map2(psi):
    '''
    Bijection, mapping each binary column of binary matrix psi onto an integer.
    '''
    ro,co=shape(psi)
    c=zeros(co)
    for t in range(co):
        for j in range(ro):
            c[t]=c[t]+psi[j,t]*(2**j)
    return c

def binTrowH(M):
    '''
    Input: Multidimensional time series M
    Output: Binarized multidimensional time series
    '''
    ro,co=shape(M)
    M2=zeros((ro,co))
    for i in range(ro):
        M2[i,:]=signal.detrend(M[i,:],axis=0)
        M2[i,:]=M2[i,:]-mean(M2[i,:])
        M3=zeros((ro,co))
        for i in range(ro):
            M2[i,:]=abs(hilbert(M2[i,:]))
            th=mean(M2[i,:])
            for j in range(co):
                if M2[i,j] >= th :
                    M3[i,j]=1
                else:
                    M3[i,j]=0
    return M3

def entropy(string):
    '''
    Calculates the Shannon entropy of a string
    '''
    string=list(string)
    prob = [ float(string.count(c)) / len(string) for c in dict.fromkeys(list(string)) ]
    entropy = - sum([ p * log(p) / log(2.0) for p in prob ])

    return entropy


def ACE(X):
    '''
    Measure ACE, using shuffled reslut as normalization.
    '''
    X=Pre(X)
    ro,co=shape(X)
    M=binTrowH(X)
    E=entropy(map2(M))
    for i in range(ro):
        shuffle(M[i])
    Es=entropy(map2(M))
    return E/float(Es)


##########
'''
SCE - Synchrony Coalition Entropy
'''
##########

def diff2(p1,p2):
    '''
    Input: two series of phases
    Output: synchrony time series thereof
    '''
    d=array(abs(p1-p2))
    d2=zeros(len(d))
    for i in range(len(d)):
        if d[i]>pi:
            d[i]=2*pi-d[i]
        if d[i]<0.8:
            d2[i]=1
    return d2


def Psi(X):
    '''
    Input: Multi-dimensional time series X
    Output: Binary matrices of synchrony for each series
    '''
    X=angle(hilbert(X))
    ro,co=shape(X)
    M=zeros((ro, ro-1, co))

    #An array containing 'ro' arrays of shape 'ro' x 'co', i.e. being the array of synchrony series for each channel.

    for i in range(ro):
        l=0
        for j in range(ro):
            if i!=j:
                M[i,l]=diff2(X[i],X[j])
                l+=1
    return M

def BinRan(ro,co):
    '''
    Create random binary matrix for normalization
    '''

    y=rand(ro,co)
    for i in range(ro):
        for j in range(co):
            if y[i,j]>0.5:
                y[i,j]=1
            else:
                y[i,j]=0
    return y

def SCE(X):
    X=Pre(X)
    ro,co=shape(X)
    M=Psi(X)
    ce=zeros(ro)
    norm=entropy(map2(BinRan(ro-1,co)))
    for i in range(ro):
        c=map2(M[i])
        ce[i]=entropy(c)

    return mean(ce)/norm,ce/norm,np.mean(M)

'''
def SCE(X):
 X=Pre(X)
 ro,co=shape(X)
 M=Psi(X)
 ce=zeros(ro)
 norm=entropy(map2(BinRan(ro-1,co)))
 for i in range(ro):
  c=map2(M[i])
  ce[i]=entropy(c)

 return mean(ce)/norm,ce/norm
 '''