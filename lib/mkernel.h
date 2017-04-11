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
    template<typename MatrixType, typename ReturnMatrixType=blaze::SymmetricMatrix<MatrixType>>
    ReturnMatrixType &operator()(MatrixType &a, ReturnMatrixType &ret) const {
        LOG_DEBUG("rows: %zu\n", a.rows());
        if(ret.rows() != a.rows()) ret.resize(a.rows());
        assert(ret.rows() == a.rows());
        assert(ret.columns() == a.rows());
        const size_t len(a.rows());
        FloatType zomg[len][len]; // VLA -- Don't use except for stupid testing!
        for(size_t i(0); i < len; ++i) {
            for(size_t j = i; j < len; ++j) {
                const auto rowi(row(a, i)), rowj(row(a, j));
                ret(i, j) = zomg[i][j] = dot(rowi, rowj) + c_;
                LOG_INFO("Cast, no storing of rows, addition: %zu, %zu value is %lf. zomg val: %lf\n", i, j, ret(i, j), zomg[i][j]);
            }
        }
        ret *= k_;
        for(size_t i(0); i < len; ++i) {
            for(size_t j = i; j < len; ++j) {
                LOG_INFO("Before second calculation: %zu, %zu value is %lf. zomg val: %lf \n", i, j, ret(i, j), zomg[i][j]);
            }
        }
        for(size_t i(0); i < len; ++i) {
            for(size_t j = i; j < len; ++j) {
                LOG_INFO("Before third calculation: %zu, %zu value is %lf. zomg val: %lf \n", i, j, ret(i, j), zomg[i][j]);
                LOG_INFO("Cast, no temporary, but with recalculation: %zu, %zu value is %lf. Second calculation: %lf\n", i, j, ret(i, j), dot(row(a, i), row(a, j)) + c_);
            }
        }
        for(size_t i(0); i < len; ++i) {
            for(size_t j = i; j < len; ++j) {
                LOG_INFO("Before tanh: %zu, %zu value is %lf. zomg val: %lf \n", i, j, ret(i, j), zomg[i][j]);
            }
        }
        ret = tanh(ret);
        for(size_t i(0); i < len; ++i) {
            for(size_t j = i; j < len; ++j) {
                LOG_INFO("After tanh: %zu, %zu value is %lf. zomg val: %lf \n", i, j, ret(i, j), zomg[i][j]);
                zomg[i][j] = ret(i, j);
                LOG_INFO("After tanh: %zu, %zu value is %lf. zomg val: %lf \n", i, j, ret(i, j), zomg[i][j]);
            }
        }
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
