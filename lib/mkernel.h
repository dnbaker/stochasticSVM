#ifndef _MKERNEL_H_
#define _MKERNEL_H_
#include "lib/kernel.h"
#include <iostream>

using std::cout;



namespace svm {

template<typename FloatType=float>
struct TanhKernelMatrix {
    const FloatType k_;
    const FloatType c_;
    template<typename MatrixType, typename ReturnMatrixType=blaze::SymmetricMatrix<MatrixType>>
    ReturnMatrixType &operator()(MatrixType &a, ReturnMatrixType &ret) const {
        if(ret.rows() != a.rows()) ret.resize(a.rows());
        assert(ret.rows() == a.rows());
        assert(ret.columns() == a.rows());
        const size_t len(a.rows());
        for(size_t i(0); i < len; ++i)
            for(size_t j = i; j < len; ++j)
                ret(i, j) = static_cast<FloatType>(dot(row(a, i), row(a, j))) + c_;
        //cerr << "pre-tanh" << ret << '\n';
        ret = tanh(ret * k_);
        //cerr << "post-tanh" << ret << '\n';
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
