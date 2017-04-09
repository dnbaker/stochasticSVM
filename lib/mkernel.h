#ifndef _MKERNEL_H_
#define _MKERNEL_H_
#include "lib/kernel.h"

namespace svm {

#if 0
template<typename FloatType>
struct LinearKernel {
    template<typename MatrixType1, typename MatrixType2>
    MatrixType1 operator()(MatrixType1 &a, MatrixType2 &b) const {
        MatrixType1 ret(a.rows(), a.columns());
        for(size_t i(0), e(rows()), i != e; ++i) {
            for(size_t i(0), e(rows()), i != e; ++i) {
                return dot<MatrixType, FloatType>(a, b);
            }
        }
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
#endif

template<typename FloatType=float>
struct TanhKernelMatrix {
    // TODO: Expand this to somehow exploit matrix structure/instrinsics for better performance?
    const FloatType k_;
    const FloatType c_;
    template<typename MatrixType>
    blaze::SymmetricMatrix<MatrixType> operator()(MatrixType &a) const {
        blaze::SymmetricMatrix<MatrixType> ret;
        for(size_t i(0); i < a.rows(); ++i) {
            for(size_t j(i); j < a.rows(); ++j) {
                ret(i, j) = dot(row(a, i), row(a, j)) + c_;
            }
        }
        ret *= k_;
        return tanh(ret);
    }
    TanhKernelMatrix(FloatType k, FloatType c): k_(k), c_(c){}
};

} // namespace svm

#endif // _MKERNEL_H_
