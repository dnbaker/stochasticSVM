#ifndef _KERNEL_H_
#define _KERNEL_H_
#include "lib/misc.h"
#include "blaze/Math.h"

namespace svm {

template<typename FloatType>
struct KernelBase {
    using float_type = FloatType;
    template<typename MatrixType>
    FloatType operator()(MatrixType &a, MatrixType &b) const;
};

template<typename FloatType, typename Kernel1, typename Kernel2>
struct MultiplicativeKernel: KernelBase<FloatType> {
    // TODO: Expand this to use fold expressions and a variable number of kernels.
    const std::pair<Kernel1, Kernel2> kernels_;
    const FloatType a_, b_;
    template<typename MatrixType>
    FloatType operator()(MatrixType &a, MatrixType &b) const {
        return a_ * std::get<0>(kernels_)(a, b) * std::get<1>(kernels_)(a, b) + b_;
    }
    MultiplicativeKernel(std::pair<Kernel1, Kernel2> &kernels,
                         FloatType a=1., FloatType b=1.): kernels_(std::move(kernels)), a_(a), b_(b) {}
};

template<typename FloatType, typename Kernel1, typename Kernel2>
struct AdditiveKernel: KernelBase<FloatType> {
    // TODO: Expand this to use fold expressions and a variable number of kernels.
    const std::pair<Kernel1, Kernel2> kernels_;
    const FloatType a_, b_, c_;
    template<typename MatrixType>
    FloatType operator()(MatrixType &a, MatrixType &b) const {
        return std::get<0>(kernels_)(a, b) * a_ + std::get<1>(kernels_)(a, b) * b_;
    }
    AdditiveKernel(std::pair<Kernel1, Kernel2> &kernels,
                   FloatType a=1., FloatType b=1., FloatType c=0.): kernels_(std::move(kernels)), a_(a), b_(b), c_(c) {}
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
        return 1. - norm2 / (norm2 + c2_);
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
struct ExpKernel: KernelBase<FloatType> {
    // TODO: Expand this to somehow exploit matrix structure/instrinsics for better performance?
    const FloatType mgamma_;
    template<typename MatrixType>
    FloatType operator()(MatrixType &a, MatrixType &b) const {
        return std::exp(mgamma_ * std::sqrt(diffnorm<MatrixType, FloatType>(a, b)));
    }
    ExpKernel(FloatType gamma): mgamma_(-gamma) {}
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
struct CauchyKernel: KernelBase<FloatType> {
    const FloatType sigma_sq_inv_;
    template<typename MatrixType>
    FloatType operator()(MatrixType &a, MatrixType &b) const {
        return 1. / (1. + diffnorm(a, b) * sigma_sq_inv_);
    }
    CauchyKernel(FloatType sigma): sigma_sq_inv_(1. / (sigma * sigma)) {}
};

template<typename FloatType>
struct ChiSqPDVariantKernel: KernelBase<FloatType> {
    template<typename MatrixType>
    FloatType operator()(MatrixType &a, MatrixType &b) const {
        return 2. * dot(a, b / (a + b));
    }
};


template<typename FloatType>
struct ChiSqKernel: KernelBase<FloatType> {
    template<typename MatrixType>
    FloatType operator()(MatrixType &a, MatrixType &b) const {
        //auto diff(a - b);
        //auto sum(a + b); Maybe do this?
        return 1. - 2. * dot(a - b, ((a - b) / (a + b)));
    }
};

template<typename FloatType>
struct StudentKernel: KernelBase<FloatType> {
    const FloatType d_;
    template<typename MatrixType>
    FloatType operator()(MatrixType &a, MatrixType &b) const {
        return 1. / (1 + std::pow(diffnorm(a, b), d_));
    }
    StudentKernel(FloatType d): d_(d / 2.) {} // Divide by 2 to get the n-nom.
};

template<typename FloatType>
struct ANOVAKernel: KernelBase<FloatType> {
    const FloatType d_, k_, sigma_;
    template<typename MatrixType>
    FloatType operator()(MatrixType &a, MatrixType &b) const {
        return std::pow(-sigma_ * diffnorm(blaze::pow(a, k_), blaze::pow(b, k_)), d_);
    }
    ANOVAKernel(FloatType d, FloatType k, FloatType sigma): d_(d / 2.), k_(k), sigma_(sigma) {}
};

template<typename FloatType>
struct DefaultWaveletFunction {
    FloatType operator()(FloatType input) {return std::cos(1.75 * input) * std::exp(input * input * -0.5);}
};

template<typename FloatType, class WaveletFunction=DefaultWaveletFunction<FloatType>>
struct WaveletKernel: KernelBase<FloatType> {
    const FloatType a_inv_, c_;
    WaveletFunction fn_;
    template<typename MatrixType>
    FloatType operator()(MatrixType &a, MatrixType &b) const {
        auto ait(a.cbegin()), bit(b.cbegin());
        FloatType ret(fn_((*ait - c_) * a_inv_) * fn_((*bit - c_) * a_inv_));
        while(++ait != a.cend() && ++bit != b.cend()) ret *= fn_((*ait - c_) * a_inv_) * fn_((*bit - c_) * a_inv_);
        return ret;
    }
    WaveletKernel(FloatType a, FloatType c): a_inv_(1. / a), c_(c) {}
};

template<typename FloatType>
struct LogarithmicKernel: KernelBase<FloatType> {
    const FloatType d_;
    template<typename MatrixType>
    FloatType operator()(MatrixType &a, MatrixType &b) const {
        return -std::log(std::pow(diffnorm(a, b), d_) + 1);
    }
    LogarithmicKernel(FloatType d): d_(d / 2.) {} // Divide by 2 to get the n-norm.
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
struct ExponentialBesselKernel: KernelBase<FloatType> {
    // https://github.com/primaryobjects/Accord.NET/blob/master/Sources/Accord.Statistics/Kernels/Bessel.cs
    // https://github.com/DiegoCatalano/Catalano-Framework/blob/master/Catalano.Statistics/src/Catalano/Statistics/Kernels/Bessel.java
    const FloatType sigma_, order_;
    template<typename MatrixType>
    FloatType operator()(MatrixType &a, MatrixType &b) const {
        const auto norm(std::sqrt(diffnorm(a, b)));
        return cyl_bessel_j(order_, sigma_ * norm) / std::exp(norm, -norm * order_);
    }
    ExponentialBesselKernel(FloatType sigma, FloatType order): sigma_(sigma), order_(order) {}
};


template<typename FloatType>
struct CylindricalBesselKernel: KernelBase<FloatType> {
    // http://crsouza.com/2010/03/17/kernel-functions-for-machine-learning-applications/#bessel
    const FloatType sigma_, vp1_, minus_nvp1_;
    template<typename MatrixType>
    FloatType operator()(MatrixType &a, MatrixType &b) const {
        const auto norm(sigma_ * std::sqrt(diffnorm(a, b)));
        return cyl_bessel_j(vp1_, sigma_ * norm) / std::exp(norm, minus_nvp1_);
    }
    CylindricalBesselKernel(FloatType sigma, FloatType n, FloatType v): sigma_(sigma), vp1_(v + 1), minus_nvp1_(-n * vp1_) {}
};

template<typename FloatType, FloatType Threshold=1e-5>
struct RBesselKernel: KernelBase<FloatType> {
    // https://github.com/cran/kernlab/blob/R/kernels.R
    const FloatType sigma_, order_, degree_, lim_;
    template<typename MatrixType>
    FloatType operator()(MatrixType &a, MatrixType &b) const {
        const auto norm(sigma_ * std::sqrt(diffnorm(a, b)));
        if(norm < Threshold) Threshold = lim_;
        FloatType tmp = norm < Threshold ? lim_: cyl_bessel_j(norm) * std::pow(norm, -order_);
        return std::pow(tmp / lim_, degree_);
    }
    RBesselKernel(FloatType sigma, int order=1, int degree=1):
        sigma_(sigma), order_(order), degree_(degree),
        lim_(1. / (std::tgamma(order_ + 1) * std::pow(2, order_))) {}
        
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
