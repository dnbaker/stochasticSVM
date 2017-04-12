# mksvm

## Description
An implementation (in-progress) of PEGASOS for both [linear](http://ttic.uchicago.edu/~shai/papers/SSSICML08.pdf) and [kernel](http://ttic.uchicago.edu/~nati/Publications/PegasosMPB.pdf) instantiations of the PEGASOS algorithm.
The linear method is notable for training-time inversely proportional to dataset size.
The kernel method runtime does increase with the number of datapoints, but unlike static solvers, the gradient descent method only has linear memory requirements.

Efforts will be made to accommodate both binary and multiclass classification.

## Dependencies
|Dependency | Reference | Comments |
|-|-|-|
|[Blaze](https://bitbucket.org/blaze-lib)|[K. Iglberger, et al.: Expression Templates Revisited: A Performance Analysis of Current Methodologies. SIAM Journal on Scientific Computing, 34(2): C42--C69, 2012](http://epubs.siam.org/sisc/resource/1/sjoce3/v34/i2/pC42_s1)|For optimal performance, this should be linked against BLAS and parallelized, as controlled in blaze/blaze/BLAS.h|
|C++14||MKSVM is currently only tested on gcc under 5.2 and 6.3|
