#ifndef _KERNEL_H_
#define _KERNEL_H_
#include "lib/misc.h"
#include "blaze/Math.h"

namespace svm {

template<typename FloatType>
struct KernelBase {
    template<typename MatrixType>
    FloatType operator()(MatrixType &a, MatrixType &b) const;
};

template<typename FloatType>
struct PolyKernel: KernelBase<FloatType> {
    const FloatType d_;
    // TODO: Expand this to somehow exploit matrix structure/instrinsics for better performance?
    template<typename MatrixType>
    FloatType operator()(MatrixType &a, MatrixType &b) const {
        const FloatType prod(dot(a, b) + d_);
        return std::pow(prod, d_);
    }
    PolyKernel(FloatType d): d_(d) {}
};

template<typename FloatType>
struct LaplacianKernel: KernelBase<FloatType> {
    // TODO: Expand this to somehow exploit matrix structure/instrinsics for better performance?
    const FloatType msigma_;
    template<typename MatrixType>
    FloatType operator()(MatrixType &a, MatrixType &b) const {
        return std::exp(msigma_ * std::sqrt(diffnorm<MatrixType, FloatType>(a, b)));
    }
    RBFKernel(FloatType sigma): msigma_(-sigma) {}
};


template<typename FloatType>
struct RBFKernel: KernelBase<FloatType> {
    // TODO: Expand this to somehow exploit matrix structure/instrinsics for better performance?
    const FloatType mgamma_;
    template<typename MatrixType>
    FloatType operator()(MatrixType &a, MatrixType &b) const {
        return std::exp(mgamma_ * diffnorm<MatrixType, FloatType>(a, b));
    }
    RBFKernel(FloatType gamma): mgamma_(-gamma) {}
};

template<typename FloatType>
struct SphericalKernel: KernelBase<FloatType> {
    const FloatType sigma_inv_;
    const FloatType sigma_;
    template<typename MatrixType>
    FloatType operator()(MatrixType &a, MatrixType &b) const {
        const FloatType norm2(std::sqrt(diffnorm(a, b)) * sigma_inv_);
        return norm2 >= 1. ? 0.
                           : 1. - 1.5 * norm2 + (norm2 * norm2 * norm2) * .5;
    }
    SphericalKernel(FloatType sigma): sigma_inv_(1 / sigma), sigma_(sigma) {}
};

template<typename FloatType>
struct CircularKernel: KernelBase<FloatType> {
    const FloatType sigma_inv_;
    const FloatType sigma_;
    static constexpr FloatType TWO_OVER_PI = 2 / M_PIl;
    template<typename MatrixType>
    FloatType operator()(MatrixType &a, MatrixType &b) const {
        const FloatType norm2(std::sqrt(diffnorm(a, b)) * sigma_inv_);
        return norm2 >= 1. ? 0.
                           : TWO_OVER_PI * (std::acos(-norm2) - norm2 * std::sqrt(1 - norm2 * norm2));
    }
    CircularKernel(FloatType sigma): sigma_inv_(1 / sigma), sigma_(sigma) {}
};

template<typename FloatType>
struct TanhKernel: KernelBase<FloatType>{
    // TODO: Expand this to somehow exploit matrix structure/instrinsics for better performance?
    const FloatType k_;
    const FloatType c_;
    template<typename MatrixType>
    FloatType operator()(MatrixType &a, MatrixType &b) const {
        return std::tanh(dot(a, b) * k_ + c_);
    }
    TanhKernel(FloatType k, FloatType c): k_(k), c_(c) {}
};

} // namespace svm

#endif // _KERNEL_H_

template<typename FloatType>
struct LinearKernel: KernelBase<FloatType> {
    // TODO: Expand this to somehow exploit matrix structure/instrinsics for better performance?
    template<typename MatrixType>
    FloatType operator()(MatrixType &a, MatrixType &b) const {
        return dot(a, b);
    }
};

template<typename FloatType>
struct LaplacianKernel: KernelBase<FloatType> {
    // TODO: Expand this to somehow exploit matrix structure/instrinsics for better performance?
    const FloatType msigma_;
    template<typename MatrixType>
    FloatType operator()(MatrixType &a, MatrixType &b) const {
        return std::exp(msigma_ * std::sqrt(diffnorm<MatrixType, FloatType>(a, b)));
    }
    RBFKernel(FloatType sigma): msigma_(-sigma) {}
};


template<typename FloatType>
struct RBFKernel: KernelBase<FloatType> {
    // TODO: Expand this to somehow exploit matrix structure/instrinsics for better performance?
    const FloatType mgamma_;
    template<typename MatrixType>
    FloatType operator()(MatrixType &a, MatrixType &b) const {
        return std::exp(mgamma_ * diffnorm<MatrixType, FloatType>(a, b));
    }
    RBFKernel(FloatType gamma): mgamma_(-gamma) {}
};

template<typename FloatType>
struct SphericalKernel: KernelBase<FloatType> {
    const FloatType sigma_inv_;
    const FloatType sigma_;
    template<typename MatrixType>
    FloatType operator()(MatrixType &a, MatrixType &b) const {
        const FloatType norm2(std::sqrt(diffnorm(a, b)) * sigma_inv_);
        return norm2 >= 1. ? 0.
                           : 1. - 1.5 * norm2 + (norm2 * norm2 * norm2) * .5;
    }
    SphericalKernel(FloatType sigma): sigma_inv_(1 / sigma), sigma_(sigma) {}
};

template<typename FloatType>
struct CircularKernel: KernelBase<FloatType> {
    const FloatType sigma_inv_;
    const FloatType sigma_;
    static constexpr FloatType TWO_OVER_PI = 2 / M_PIl;
    template<typename MatrixType>
    FloatType operator()(MatrixType &a, MatrixType &b) const {
        const FloatType norm2(std::sqrt(diffnorm(a, b)) * sigma_inv_);
        return norm2 >= 1. ? 0.
                           : TWO_OVER_PI * (std::acos(-norm2) - norm2 * std::sqrt(1 - norm2 * norm2));
    }
    CircularKernel(FloatType sigma): sigma_inv_(1 / sigma), sigma_(sigma) {}
};

template<typename FloatType>
struct TanhKernel: KernelBase<FloatType>{
    // TODO: Expand this to somehow exploit matrix structure/instrinsics for better performance?
    const FloatType k_;
    const FloatType c_;
    template<typename MatrixType>
    FloatType operator()(MatrixType &a, MatrixType &b) const {
        return std::tanh(dot(a, b) * k_ + c_);
    }
    TanhKernel(FloatType k, FloatType c): k_(k), c_(c) {}
};

} // namespace svm

#endif // _KERNEL_H_
