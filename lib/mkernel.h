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
    const FloatType k_;
    const FloatType c_;
    template<typename MatrixType>
    blaze::SymmetricMatrix<MatrixType> &operator()(MatrixType &a, blaze::SymmetricMatrix<MatrixType> ret) const {
        if(ret.rows() != a.rows()) ret.resize(a.rows());
        assert(ret.rows() == a.rows());
        assert(ret.columns() == a.rows());
        ret(3, 2) = 4.;
        LOG_DEBUG("Ret of 3,2: %f\n", ret(3, 2));
        for(size_t i(0); i < a.rows(); ++i) {
            for(size_t j(i); j < a.rows(); ++j) {
                ret(i, j) = dot(row(a, i), row(a, j)) + c_;
                LOG_DEBUG("At %zu, %zu value is %f. Dot: %f. c: %f. Should be %f\n", i, j, ret(i, j), dot(row(a, i), row(a, j)), c_, dot(row(a, i), row(a, j)) + c_);
            }
        }
        ret *= k_;
        ret = tanh(ret);
        return ret;
    }
    template<typename MatrixType>
    blaze::SymmetricMatrix<MatrixType> operator()(MatrixType &a) const {
        blaze::SymmetricMatrix<MatrixType> ret(a.rows());
        operator()<MatrixType>(a, ret);
    }
    TanhKernelMatrix(FloatType k, FloatType c): k_(k), c_(c){}
};

} // namespace svm

#endif // _MKERNEL_H_
