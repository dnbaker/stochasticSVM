#ifndef _META_KERNEL_H
#define _META_KERNEL_H
#include "lib/kernel.h"

namespace svm {

template<typename FloatType, typename Kernel1, typename Kernel2>
struct MultiplicativeKernel: KernelBase<FloatType> {
    const std::pair<Kernel1, Kernel2> kernels_;
    const FloatType a_, b_;
    template<typename MatrixType1, typename MatrixType2>
    FloatType operator()(const MatrixType1 &a, const MatrixType2 &b) const {
        return a_ * kernels_.first(a, b) * kernels_(a, b) + b_;
    }
    MultiplicativeKernel(std::pair<Kernel1, Kernel2> kernels,
                         FloatType a=1., FloatType b=1.): kernels_(kernels), a_(a), b_(b) {}
    MultiplicativeKernel(Kernel1 k1, Kernel2 k2,
                         FloatType a=1., FloatType b=1.): MultiplicativeKernel{std::make_pair(k1, k2), a, b} {}
    std::string str() const {
        return std::string("MultiplicativeKernel:[") + std::to_string(a_) +
            ", " + std::to_string(b_) + "]{" +
            kernels_.first.str() + ", " + kernels_.second.str() + '}';
    }
};

template<typename FloatType, typename Kernel1, typename Kernel2>
struct AdditiveKernel: KernelBase<FloatType> {
    Kernel1 kernel1;
    Kernel2 kernel2;
    const FloatType a_, b_;
    template<typename MatrixType1, typename MatrixType2>
    FloatType operator()(const MatrixType1 &a, const MatrixType2 &b) const {
        return kernel1(a, b) * a_ +kernel2(a, b) * b_;
    }
    AdditiveKernel(Kernel1 kernel1, Kernel2 kernel2, FloatType a=1., FloatType b=1.):
        kernel1{kernel1}, kernel2{kernel2}, a_(a), b_(b) {}
    std::string str() const {
        return std::string("AdditiveKernel:[") + std::to_string(a_) +
            ", " + std::to_string(b_) + "]{" +
            kernel1.str() + ", " + kernel2.str() + '}';
    }
    template<typename RowType>
    INLINE void sample(RowType &row) const {
        RowType tmp(row.size());
        kernel1.sample(tmp);
        kernel2.sample(row);
        row = row * b_ + tmp * a_;
    }
};



} // namespace svm

#endif // #ifndef _META_KERNEL_H
