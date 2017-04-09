#ifndef _MKERNEL_H_
#define _MKERNEL_H_

namespace svm {

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

#endif // _MKERNEL_H_
