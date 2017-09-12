#ifndef _MKSVM_MATH_H__
#define _MKSVM_MATH_H__
#include "lib/misc.h"

namespace svm {

template<class MatrixType>
auto frobenius_norm(const MatrixType &matrix) {
    using FloatType = typename MatrixType::ElementType;
    FloatType ret(0.);
    #pragma omp parallel reduction(+:ret)
    for(size_t i = 0, e = matrix.rows(); i < e; ++i) {
        auto mrow(row(matrix, i));
        ret += dot(mrow, mrow);
    }
    return std::sqrt(ret);
}

}

#endif // #ifndef _MKSVM_MATH_H__
