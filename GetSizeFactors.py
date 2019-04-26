#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Feb 19 08:14:31 2019

@author: Alma Andersson

"""

import numpy as np
from scipy.stats import nbinom
from scipy.optimize import lsq_linear

def GetSizeFactors(cmat : np.ndarray,
                   pool_size : int = 10,
                   ) -> np.ndarray:
    
    """
    Estimates size factors to scale observed data
    Primarly inspired by Lun.et alls method proposed in 
    
    "Pooling across cells to normalize
    single-cell RNA sequencing data with many
    zero counts"
    
    doi: 10.1186/s13059-016-0947-7
    
    But with some minor adjustments
    
    With:
        Y_ij - observed counts
        Z_ij - counts adjusted for cell-bias
        s_j  - cell specific bias 
        t_j  - cell specific scaling factor
        l_i0 - expected count for gene i (non-DGE)
    Then:
        E[Y_ij]     = l_i0 * s_j
        Z _ij       = Y_ij / t_j
        E[Z_ij]     = l_i0 * s_j / t_j
    
    Function returns s_j for each cell, thus to
    obtain the adjusted counts divide the raw counts
    by the size factors.
    
    
    Parameters :
        cmat    - count matrix (n_samples, n_genes)
        pool_size  - set size of each pool, default 10 
    
    Returns :
        size_factors - cell specific size factors (n_samples,)
    
    """
    
    def map1d(i,j,m):
        """
        Maps between 1d and 2d array
        use for fast assembly
        """
        return i*m + j

    nsamples = cmat.shape[0]
    # compute library size for each sample
    libsize = np.sum(cmat,axis = 1)
    
    # sort indices according to library size
    ordr = np.argsort(libsize)[::-1]
    
    # make circular index list 
    newidx = np.zeros(ordr.shape)
    even = np.arange(0,newidx.shape[0],2)
    odd = np.arange(1,newidx.shape[0],2)
    newidx[0:even.shape[0]] = ordr[even]
    newidx[even.shape[0]:(even.shape[0] + odd.shape[0])] = ordr[np.flip(odd)]
    newidx = newidx.astype(np.int)
    
    # create sets of pool indices
    pools = np.vstack([newidx[((x + np.arange(pool_size)) % \
                              libsize.shape[0]).astype(np.int)] \
                              for x in range(libsize.shape[0])])
    
    # map indices to 1d vector coordinates for 
    # fast assembly of memberships matrix
    pools1d = map1d(np.arange(pools.shape[0]).reshape(-1,1),
                    pools,
                    nsamples)
    
    # Adjusted Expression Values
    Z = cmat / libsize.reshape(-1,1)
    
    # Summed Adjusted Expression Value for each gene over each pool
    V = np.vstack([Z[pools[k,:],:].sum(axis = 0) \
                   for k in range(pools.shape[0])])
    
    # Pseudo cell expression profile for each gene
    U = cmat.mean(axis=0).reshape(1,-1)
    
    # Ratio of Pools and Reference 
    R = V / U
    
    # Membership matrix for lstq
    A = np.zeros((pools.shape[0]*nsamples))
    A[pools1d] = 1.0
    # reshape memership vector into 2d matrix
    A = A.reshape((pools.shape[0],nsamples))
    #Matrix of estimatead theta/libsize values
    T = lsq_linear(A,
                   R.mean(axis=1),
                   bounds = (0,np.inf)).x

    # adjust for library size parameter
    S = T * libsize

    return S

if __name__ == '__main__':
    
    import time
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as font_manager
    plt.style.use('dark_background')
    
    # Simulate Dataset of non DE genes --------------------
    
    nsamples = 100
    ngenes = 2000
    # parameters for nb distribution
    n = np.repeat(np.random.randint(1,10,ngenes).reshape(1,-1),nsamples,axis = 0)
    p = np.repeat(np.random.uniform(0,1,ngenes).reshape(1,-1),nsamples, axis = 0)
    # scaling factors for rate
    theta = np.random.uniform(0,20,n.shape[0]).reshape(-1,1)
    # scaled rate parameters
    ns = n * theta

    # generate synthetic count matrix
    cmat = nbinom.rvs(n = ns,p = p)
    # remove bas samples and genes
    cmat = cmat[cmat.sum(axis = 1) > 50, :]
    cmat = cmat[:,cmat.sum(axis = 0) > 50]
    # define pool_size size
    pool_size = 10
    
    # Estimate size_factors and time procedure
    start = time.time()
    size_factors = GetSizeFactors(cmat,pool_size)
    end = time.time()
    print(f"Elapsed time >> {end-start}")
    
    
    # Visualization --------------
    
    fpth = list(filter( lambda x : 'texgyreadventor-bold.otf' in x, \
                       font_manager.findSystemFonts(fontpaths=None, fontext='ttf')))
    
    prop =  font_manager.FontProperties(fname = (fpth[0] if len(fpth) > 0 else 'sans'))
    
    titlestyle = dict(fontproperties = prop,
                      fontsize = 15)
    
    imstyle = dict(aspect = 'auto',
                   cmap = plt.cm.magma)
    
    plt.figure(figsize =  (18,5))
    # unadjusted
    plt.grid(b = True, axis = 'both')
    
    plt.subplot(1,3,1)
    plt.title('Unadjusted Count Matrix',
              **titlestyle)
    
    plt.imshow(np.log1p(cmat),
               **imstyle,
               )
    plt.xlabel('gene')
    plt.ylabel('sample')

    # scaled 
    plt.subplot(1,3,2)
    plt.title('Scaled Count Matrix using size factors', 
              **titlestyle)
    plt.imshow(np.log1p(cmat / size_factors.reshape(-1,1)),
               **imstyle,
               )
    plt.xlabel('gene')
    plt.ylabel('sample')
    
    plt.subplot(1,3,3)
    plt.title('true factors vs. estimated factors',
              **titlestyle)
    
    plt.scatter(theta,
                size_factors,
                )
    plt.xlabel('true factor',
               **titlestyle)
    plt.ylabel('estimated factors',
               **titlestyle)
    