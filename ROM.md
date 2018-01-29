# Orthogonal Random Projections

[Primary Reference:  The Unreasonable Effectiveness of Structured Random Orthogonal Embeddings](https://arxiv.org/abs/1703.00864)
[First paper to demonstrate improved accuracy using structure random embeddings: Yu, et al. 2016](https://arxiv.org/abs/1610.09072)


Definitions
===========
1. G-ort (Orthogonal)
  1. Random Gaussian matrix conditioned on rows being orthogonal.
  2. SD-product matrix.
    1. Formed by multiplying some number k of SD blocks, each of which is highly structured, leading to fast computation of products.
    2. S is a _S_tructured Matrix
    3. D is a random _D_iagonal matrix.
    4. Cf. Fastfood, "Key to Fastfood is the observation that Hadamard matrices, when combined with diagonal Gaussian matrices exhibit properties similar to dense Gaussian matrices. Yet unlike the latter, Hadamard and diagonal matrices are inexpensive to multiply and store."
    5. The SD block generalizes the HD block.

The first paper extended its usefulness to kernels besides Gaussian, explains the effectiveness of multiple [HS]D blocks.

Applications
===========
1. Johnson-Lindenstrauss Transform (JL transform) for dimensionality reduction.
  2. Orthogonal JL transform
    1. Strictly smaller MSE than unstructured JL transform and is as fast as the fast JL transform methods.
    1. SD-block methods outperform even G-ort methods, but a k = 1 performs better than k > 1.
    1. MSR is the real-only method. (Structured-Rademacher random matrix with k \in N blocks.) Extending to cmplex numbers significantly improves the OJLT.
      1. I wonder about the value of this. It seems to provide a doubling of randomness ("two" numbers per number, and error is inversely proportional to m.)
      2. If there's a way to do this without doubling the memory of all downstream operations, it could be an improvement. Otherwise, we're just effectively doubling m.
2. Kernel methods with random feature maps. (e.g., Random Fourier Features).
  1. Guaranteed better MSE.
  2. SD-block methods outperform even G-ort methods. odd numbers are best. The error dependent on k decays exponentially, explaining diminishing returns for k = (i * 2 + 1) for i > 1.
3. Gram matrix approximation.
4. My thoughts
  1. Could this be applied to Bayesian optimization? [Ref](https://arxiv.org/pdf/1301.1942.pdf)
  2. Approximate matrix decompositions? [Ref](https://arxiv.org/abs/0909.4061)
  3. Kmeans clustering? [Ref](https://arxiv.org/abs/1011.4632)
  4. Gaussian Mixture Models
    1. [Dasgupta](cseweb.ucsd.edu/~dasgupta/papers/mog.pdf), [Dasgupta, et al.](https://arxiv.org/abs/0912.0086)
  5. Signal reconstruction? [Ref](http://people.ece.umn.edu/~miha../jdhaupt/publications/it06_noisy_random_proj.pdf)
  6. Could this allow for kernelizing GMMs?


Implementation
=============
1. G-ort
  1. Description
    1. G (plain) is an R^(mn) matrix consisting of iid values from N(0, 1) [zero mean, unit variance].
    2. "May be viewed as the regular conditional propbability measure associated with G considioned on the event that all rows of the matrix are orthogonal."
  1. Method
    1. First, generate a plain G.
    2. Perform a Gram-Schmidt process on the rows of G to obtain a set of orthonormal rows.
    3. Randomly, independently scale each row so that "margically" it has the distribution of a Gaussian vector.
  2. Note: takes large amounts of space and significant time for multiplication.
  3. This is a specialization of the random matrix used for random fourier features, for example.
2. SD-product matrices.
  1. For S, we naturally use the normalized Hadamard matrix in our implementation due to its highly optimized calculation using the FFT. (see FFTW or FFHT, the one we use.)
  2. For MSR (the real form), for all D_R are rademacher random variables on the diagonals.
    1. That could make it possible to store with 1 bit per diagonal element. This could benefit from an optimized, specialized FHT for packed usage.
    2. That also means that the complex variant is much more expensive, both in terms of memory and operations.
  3. When using, we subsample m random rows without replacement from the left-most matrix.


