# stochasticSVM

## Description
An implementation of PEGASOS, powered by [Blaze](https://bitbucket.org/blaze-lib).

Complete: linear [[reference]](http://ttic.uchicago.edu/~shai/papers/SSSICML08.pdf). kernel. [[reference]](http://ttic.uchicago.edu/~nati/Publications/PegasosMPB.pdf).

Incomplete: Randomized Fourier Feature and Forgetron/Budgeted kernel algorithms.

The linear method is notable for training-time with inverse dependence on dataset size.
The kernel method runtime does increase with the number of datapoints, but unlike static solvers, the gradient descent method only has linear memory requirements.

Efforts will eventually be made to accommodate multiclass classification, though binary is the only form currently supported.

Data may be provided in either sparse SVM form (SVM-light format) or a dense tab-delimited format.

## Dependencies
|Dependency | Reference | Comments |
|-|-|-|
|[Blaze](https://bitbucket.org/blaze-lib)|[K. Iglberger, et al.: Expression Templates Revisited: A Performance Analysis of Current Methodologies. SIAM Journal on Scientific Computing, 34(2): C42--C69, 2012](http://epubs.siam.org/sisc/resource/1/sjoce3/v34/i2/pC42_s1)|For optimal performance, this should be linked against BLAS and parallelized, as controlled in blaze/blaze/config/BLAS.h|
|C++14||DenseSVM is currently only tested on gcc under 5.2 and 6.3|
|OpenMP||OpenMP is currently required for certain operations, but this requirement could be easily removed.|

Blaze leverages SIMD, BLAS, and expression template metaprogramming for state-of-the-art linear algebra performance.

## Building

Simply ``make``.

If you wish to use floats instead of doubles, which can be twice as fast in arithmetic operations due to SIMD, use ``make FLOAT_TYPE=float``.

## TODO

0. Add simple streaming classifier using already-built SVM.
1. Expand to multiclass.
2. Add Random Fourier Feature extractor.
  1. Add Hadamard/Fourier "FastFood" adaptor.
  2. Consider attempting to outperform F2F.
3. Consider compressive projections.
