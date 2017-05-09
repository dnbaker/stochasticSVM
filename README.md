# denseSVM

## Description
An implementation of PEGASOS, powered by [Blaze]https://github.com/dnbh/denseSVM.

Complete: linear [[reference]](http://ttic.uchicago.edu/~shai/papers/SSSICML08.pdf).

Incomplete: kernel. [[reference]](http://ttic.uchicago.edu/~nati/Publications/PegasosMPB.pdf).

The linear method is notable for training-time with inverse dependence on dataset size.
The kernel method runtime does increase with the number of datapoints, but unlike static solvers, the gradient descent method only has linear memory requirements.

Efforts will eventually be made to accommodate multiclass classification, though binary is the only form currently supported.

Data may be provided in either sparse SVM form (SVM-light format) or a dense tab-delimited format.

## Dependencies
|Dependency | Reference | Comments |
|-|-|-|
|[Blaze](https://bitbucket.org/blaze-lib)|[K. Iglberger, et al.: Expression Templates Revisited: A Performance Analysis of Current Methodologies. SIAM Journal on Scientific Computing, 34(2): C42--C69, 2012](http://epubs.siam.org/sisc/resource/1/sjoce3/v34/i2/pC42_s1)|For optimal performance, this should be linked against BLAS and parallelized, as controlled in blaze/blaze/BLAS.h|
|C++14||DenseSVM is currently only tested on gcc under 5.2 and 6.3|
|OpenMP|OpenMP is currently required for certain operations, but this requirement could be easily removed.|
