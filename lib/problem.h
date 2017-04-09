#ifndef _PROBLEM_H_
#define _PROBLEM_H_
#include "lib/misc.h"
#include "blaze/Math.h"
#include "lib/parse.h"

namespace svm {


// TODO: Polynomial kernels

template<typename MatrixType, typename FloatType=float>
INLINE dot(MatrixType &a, MatrixType &b) {
    return a * trans(b);
}

template<typename MatrixType, typename FloatType=float>
INLINE diffnorm(MatrixType &a, MatrixType &b) {
    const auto norm(a - b);
    return dot(norm, norm);
}

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

template<class Kernel, typename MatrixType=float, typename VectorType=int>
class SVM {
    DynamicMatrix<MatrixType> m_;
    DynamicVector<VectorType> v_;
    SVM(const char *path) {
        std::tie(m_, v_) = parse_problem<MatrixType, VectorType>(path);
    }
};

} // namespace svm


#endif // _PROBLEM_H_
