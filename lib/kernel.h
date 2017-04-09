#ifndef _KERNEL_H_
#define _KERNEL_H_
#include "lib/misc.h"
#include "blaze/Math.h"
#include "lib/parse.h"

namespace svm {

template<typename FloatType>
struct LinearKernel {
    // TODO: Expand this to somehow exploit matrix structure/instrinsics for better performance?
    template<typename MatrixType>
    FloatType operator()(MatrixType &a, MatrixType &b) const {
        return dot<MatrixType, FloatType>(a, b);
    }
};

template<typename FloatType>
struct RBFKernel {
    // TODO: Expand this to somehow exploit matrix structure/instrinsics for better performance?
    const FloatType mgamma_;
    template<typename MatrixType>
    FloatType operator()(MatrixType &a, MatrixType &b) const {
        return std::exp(mgamma_ * diffnorm<MatrixType, FloatType>(a, b));
    }
    RBFKernel(FloatType gamma): mgamma_(-gamma) {}
};

template<typename FloatType>
struct TanhKernel {
    // TODO: Expand this to somehow exploit matrix structure/instrinsics for better performance?
    const FloatType k_;
    const FloatType c_;
    template<typename MatrixType>
    FloatType operator()(MatrixType &a, MatrixType &b) const {
        return std::tanh(dot<MatrixType, FloatType>(a, b) * k_ + c_) ;
    }
    TanhKernel(FloatType k, FloatType c): k_(k), c_(c) {}
};

} // namespace svm

#endif // _KERNEL_H_
