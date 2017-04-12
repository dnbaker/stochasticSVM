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
struct LinearKernel: KernelBase<FloatType> {
    // TODO: Expand this to somehow exploit matrix structure/instrinsics for better performance?
    template<typename MatrixType>
    FloatType operator()(MatrixType &a, MatrixType &b) const {
        return dot(a, b);
    }
};

template<typename FloatType>
struct PolyKernel: KernelBase<FloatType> {
    const FloatType a_;
    const FloatType d_;
    // TODO: Expand this to somehow exploit matrix structure/instrinsics for better performance?
    template<typename MatrixType>
    FloatType operator()(MatrixType &a, MatrixType &b) const {
        const FloatType prod(a * dot(a, b) + d_);
        return std::pow(prod, d_);
    }
    PolyKernel(FloatType a, FloatType d): a_(a), d_(d) {}
};

template<typename FloatType>
struct LaplacianKernel: KernelBase<FloatType> {
    // TODO: Expand this to somehow exploit matrix structure/instrinsics for better performance?
    const FloatType msigma_;
    template<typename MatrixType>
    FloatType operator()(MatrixType &a, MatrixType &b) const {
        return std::exp(msigma_ * std::sqrt(diffnorm<MatrixType, FloatType>(a, b)));
    }
    LaplacianKernel(FloatType sigma): msigma_(-sigma) {}
};


template<typename FloatType>
struct RationalQuadKernel: KernelBase<FloatType> {
    // TODO: Expand this to somehow exploit matrix structure/instrinsics for better performance?
    const FloatType c2_;
    template<typename MatrixType>
    FloatType operator()(MatrixType &a, MatrixType &b) const {
        const FloatType norm2(diffnorm(a, b));
        return 1 - norm2 / (norm2 + c2_);
    }
    RationalQuadKernel(FloatType c): c2_(c * c) {}
};


template<typename FloatType>
struct MultiQuadKernel: KernelBase<FloatType> {
    // TODO: Expand this to somehow exploit matrix structure/instrinsics for better performance?
    const FloatType c2_;
    template<typename MatrixType>
    FloatType operator()(MatrixType &a, MatrixType &b) const {
        const FloatType norm2(diffnorm(a, b));
        return std::sqrt(norm2 + c2_);
    }
    MultiQuadKernel(FloatType c): c2_(c * c) {}
};

template<typename FloatType>
struct InvMultiQuadKernel: KernelBase<FloatType> {
    // TODO: Expand this to somehow exploit matrix structure/instrinsics for better performance?
    const FloatType c2_;
    template<typename MatrixType>
    FloatType operator()(MatrixType &a, MatrixType &b) const {
        const FloatType norm2(diffnorm(a, b));
        return 1. / std::sqrt(norm2 + c2_);
    }
    InvMultiQuadKernel(FloatType c): c2_(c * c) {}
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
struct GeneralHistogramKernel: KernelBase<FloatType> {
    const FloatType a_;
    const FloatType b_;
    template<typename MatrixType>
    FloatType operator()(MatrixType &a, MatrixType &b) const {
        if(a.size() != b.size())
            throw std::out_of_range(std::string("Could not calculate min between arrays of different sizes (") + std::to_string(a.size()) + ", " + std::to_string(b.size()) + ")");
        auto ait(a.cbegin()), bit(b.cbegin());
        FloatType ret(std::min(*ait, *bit));
        while(++ait != a.cend()) ret += std::min(std::abs(std::pow(*ait, a_)), std::abs(std::pow(*++bit, b_)));
        return ret;
    }
    GeneralHistogramKernel(FloatType a, FloatType b): a_(a), b_(b) {}
};


template<typename FloatType>
struct HistogramKernel: KernelBase<FloatType> {
    template<typename MatrixType>
    FloatType operator()(MatrixType &a, MatrixType &b) const {
        if(a.size() != b.size())
            throw std::out_of_range(std::string("Could not calculate min between arrays of different sizes (") + std::to_string(a.size()) + ", " + std::to_string(b.size()) + ")");
        auto ait(a.cbegin()), bit(b.cbegin());
        FloatType ret(std::min(*ait, *bit));
        while(++ait != a.cend()) ret += std::min(*ait, *++bit);
        return ret;
    }
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
