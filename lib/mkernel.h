#ifndef _MKERNEL_H_
#define _MKERNEL_H_
#include "lib/kernel.h"
#include <iostream>

using std::cout;



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
    template<typename MatrixType, typename ReturnMatrixType=blaze::SymmetricMatrix<MatrixType>>
    ReturnMatrixType &operator()(MatrixType &a, ReturnMatrixType &ret) const {
        cerr << "rows: " << a.rows() << '\n';
        if(ret.rows() != a.rows()) ret.resize(a.rows());
        assert(ret.rows() == a.rows());
        assert(ret.columns() == a.rows());
        const size_t len(a.rows());
        for(size_t i(0); i < len; ++i) {
            for(size_t j = i; j < len; ++j) {
                ret(i, j) = static_cast<FloatType>(dot(row(a, i), row(a, j))) + c_;
                std::fprintf(stderr, "[%s] Value at %zu, %zu is %f\n", __func__, i, j, ret(i, j));
                cerr << "Value at " << i << ", " << j << " is " << ret(i, j) << '\n';
            }   
        }
        cerr << "ret is \n" << ret;
        ret = tanh(ret * k_);
        cerr << "after tanh\n" << ret;
        return ret;
    }
    template<typename MatrixType, typename ReturnMatrixType=blaze::SymmetricMatrix<MatrixType>>
    ReturnMatrixType operator()(MatrixType &a) const {
        ReturnMatrixType ret(a.rows());
        operator()<MatrixType>(a, ret);
        return ret;
    }
    TanhKernelMatrix(FloatType k, FloatType c): k_(k), c_(c){}
};

} // namespace svm

#endif // _MKERNEL_H_
