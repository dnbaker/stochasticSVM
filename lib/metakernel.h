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
        return a_ * std::get<0>(kernels_)(a, b) * std::get<1>(kernels_)(a, b) + b_;
    }
    MultiplicativeKernel(std::pair<Kernel1, Kernel2> &kernels,
                         FloatType a=1., FloatType b=1.): kernels_(std::move(kernels)), a_(a), b_(b) {}
    std::string str() const {
        return std::string("MultiplicativeKernel:[") + std::to_string(a_) +
            ", " + std::to_string(b_) + "]{" +
            kernels_.first.str() + ", " + kernels_.second.str() + '}';
    }
};

template<typename FloatType, typename Kernel1, typename Kernel2>
struct AdditiveKernel: KernelBase<FloatType> {
    const std::pair<Kernel1, Kernel2> kernels_;
    const FloatType a_, b_, c_;
    template<typename MatrixType1, typename MatrixType2>
    FloatType operator()(const MatrixType1 &a, const MatrixType2 &b) const {
        return std::get<0>(kernels_)(a, b) * a_ + std::get<1>(kernels_)(a, b) * b_;
    }
    AdditiveKernel(std::pair<Kernel1, Kernel2> &kernels,
                   FloatType a=1., FloatType b=1., FloatType c=0.):
        kernels_(std::move(kernels)), a_(a), b_(b), c_(c) {}
    AdditiveKernel(Kernel1 kernel1, Kernel2 kernel2, FloatType a=1., FloatType b=1., FloatType c=0.):
        AdditiveKernel(make_pair(kernel1, kernel2), a, b, c) {}
    std::string str() const {
        return std::string("AdditiveKernel:[") + std::to_string(a_) +
            ", " + std::to_string(b_) + ", " + std::to_string(c_) + "]{" +
            kernels_.first.str() + ", " + kernels_.second.str() + '}';
    }
};



} // namespace svm

#endif // #ifndef _META_KERNEL_H
