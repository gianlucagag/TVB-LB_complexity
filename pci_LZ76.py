import numpy as np
import scipy
import scipy.sparse 

import LZ76
    
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
    return Hsrc,complexity/norm_factor
