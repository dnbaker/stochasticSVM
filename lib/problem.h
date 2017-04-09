#ifndef _PROBLEM_H_
#define _PROBLEM_H_
#include "lib/misc.h"
#include "blaze/Math.h"
#include "lib/parse.h"
#include "lib/kernel.h"

namespace svm {


// TODO: Polynomial kernels
// TODO: Gradients


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
