#ifndef _PROBLEM_H_
#define _PROBLEM_H_
#include "lib/misc.h"
#include "blaze/Math.h"
#include "lib/parse.h"

namespace svm {

template<typename FloatType>
struct LinearKernel {
    template<typename MatrixType>
    FloatType operator()(MatrixType &a, MatrixType &b) const {
        return a * trans(b);
    }
};

template<typename FloatType>
struct RBFKernel {
    const FloatType mgamma_;
    template<typename MatrixType>
    FloatType operator()(MatrixType &a, MatrixType &b) const {
        const auto diff(a - b);
        const FloatType norm(diff * trans(diff));
        return std::exp(mgamma_ * norm);
    }
    RBFKernel(FloatType gamma): mgamma_(-gamma) {}
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
