## Method To Scale Sparse Count Data

Estimates size factors to scale observed data

Primarly inspired by Lun.et alls method proposed in 
    
"Pooling across cells to normalize single-cell RNA sequencing data with many zero counts"

doi: 10.1186/s13059-016-0947-7
    
Some adjustments to the original method have been added, mainly for improved efficiency

* The linear equation system is solved using a uniformly weighted and  bounded LSQ method where no extra rows are added to the system
* Rather than using NNLS for each gene and then estimate the median value. Estimates of the _mean_ scaling factors over all genes are made in a single fitting procedure

Note how both methods rely heavily on the fact that the _majority_ of the genes are not DE within the sample. Lun et al. suggest clustering of cells with similar expressions and then estimate the size
factors conditioned on each cluster. This is **not** implemented in this method, but rather something the user must amend to him/herself.

Example of results

![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Example with simulated data")
    
