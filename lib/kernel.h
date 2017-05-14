#ifndef _KERNEL_H_
#define _KERNEL_H_
#include "lib/misc.h"
#include "blaze/Math.h"

namespace svm {

template<typename FloatType>
struct KernelBase {
    using float_type = FloatType;
    template<typename MatrixType1, typename MatrixType2>
    FloatType operator()(const MatrixType1 &a, const MatrixType2 &b) const;
    std::string str() const {
        throw std::runtime_error("NotImplementedError");
        return "KernelBase";
    }
};

template<typename FloatType>
struct LinearKernel: KernelBase<FloatType> {
    template<typename MatrixType1, typename MatrixType2>
    FloatType operator()(const MatrixType1 &a, const MatrixType2 &b) const {
        return dot(a, b);
    }
    LinearKernel() {}
    std::string str() const {return "LinearKernel";}
};

template<typename FloatType>
struct PolyKernel: KernelBase<FloatType> {
    const FloatType a_;
    const FloatType d_;
    template<typename MatrixType1, typename MatrixType2>
    FloatType operator()(const MatrixType1 &a, const MatrixType2 &b) const {
        const FloatType prod(a * dot(a, b) + d_);
        return std::pow(prod, d_);
    }
    PolyKernel(FloatType a, FloatType d): a_(a), d_(d) {}
    std::string str() const {
        return std::string("PolyKernel:{") + std::to_string(a_) + ", " +
                           std::to_string(d_) + "}";
    }
};

template<typename FloatType>
struct LaplacianKernel: KernelBase<FloatType> {
    const FloatType msigma_;
    template<typename MatrixType1, typename MatrixType2>
    FloatType operator()(const MatrixType1 &a, const MatrixType2 &b) const {
        return std::exp(msigma_ * std::sqrt(
            diffnorm<MatrixType1, MatrixType2, FloatType>(a, b)));
    }
    LaplacianKernel(FloatType sigma): msigma_(-sigma) {}
    std::string str() const {
        return std::string("LaplacianKernel:{") +
            std::to_string(-msigma_) + '}';
    }
};

template<typename FloatType>
struct MultiQuadKernel: KernelBase<FloatType> {
    const FloatType sigma_sq_, factor_, alpha_;
    template<typename MatrixType1, typename MatrixType2>
    FloatType operator()(const MatrixType1 &a, const MatrixType2 &b) const {
        return sigma_sq_ * std::pow(1 + std::sqrt(diffnorm(a, b)) * factor_, -alpha_);
    }
    MultiQuadKernel(FloatType sigma, FloatType alpha, FloatType ell):
        sigma_sq_(sigma * sigma), factor_(1./(2 * alpha * ell * ell)), alpha_(alpha) {}
    std::string str() const {
        return std::string("MultiQuadKernel:{") +
            std::to_string(std::sqrt(sigma_sq_)) + ", " +
            std::to_string(std::sqrt(.5 * alpha_ /(factor_))) +
            ", " + std::to_string(alpha_) + '}';
    }
};

template<typename FloatType>
struct SimpleQuadKernel: KernelBase<FloatType> {
    const FloatType sigma_sq_, factor_;
    template<typename MatrixType1, typename MatrixType2>
    FloatType operator()(const MatrixType1 &a, const MatrixType2 &b) const {
        return sigma_sq_ * (1 + std::sqrt(norm2(diffnorm(a, b))) * factor_);
    }
    SimpleQuadKernel(FloatType sigma, FloatType ell, FloatType):
        sigma_sq_(sigma * sigma), factor_(1./(2 * ell * ell)) {}
    std::string str() const {
        return std::string("SimpleQuadKernel:{") +
            std::to_string(std::sqrt(sigma_sq_)) + ", " +
            std::to_string(std::sqrt(.5 /(factor_))) + '}';
    }
};

template<typename FloatType>
struct InvMultiQuadKernel: KernelBase<FloatType> {
    const FloatType c2_;
    template<typename MatrixType1, typename MatrixType2>
    FloatType operator()(const MatrixType1 &a, const MatrixType2 &b) const {
        
    }
    InvMultiQuadKernel(FloatType c): c2_(c * c) {}
    std::string str() const {
        return std::string("InvMultiQuadKernel:{") +
            std::to_string(std::sqrt(c2_)) + '}';
    }
};

template<typename FloatType>
struct RBFKernel: KernelBase<FloatType> {
    const FloatType mgamma_;
    template<typename MatrixType1, typename MatrixType2>
    FloatType operator()(const MatrixType1 &a, const MatrixType2 &b) const {
        return std::exp(mgamma_ *
                        diffnorm<MatrixType1, MatrixType2, FloatType>(a, b));
    }
    RBFKernel(FloatType gamma): mgamma_(-gamma) {}
    std::string str() const {
        return std::string("RBFKernel:{") + std::to_string(-mgamma_) + '}';
    }
};

template<typename FloatType>
struct ExpKernel: KernelBase<FloatType> {
    const FloatType mgamma_;
    template<typename MatrixType1, typename MatrixType2>
    FloatType operator()(const MatrixType1 &a, const MatrixType2 &b) const {
        return std::exp(mgamma_ * std::sqrt(
            diffnorm<MatrixType1, MatrixType2, FloatType>(a, b)));
    }
    ExpKernel(FloatType gamma): mgamma_(-gamma) {}
    std::string str() const {
        return std::string("ExpKernel:{") + std::to_string(mgamma_) + '}';
    }
};

template<typename FloatType>
struct SphericalKernel: KernelBase<FloatType> {
    const FloatType sigma_inv_;
    const FloatType sigma_;
    template<typename MatrixType1, typename MatrixType2>
    FloatType operator()(const MatrixType1 &a, const MatrixType2 &b) const {
        const FloatType norm2(std::sqrt(diffnorm(a, b)) * sigma_inv_);
        return norm2 >= 1. ? 0.
                           : 1. - 1.5 * norm2 + (norm2 * norm2 * norm2) * .5;
    }
    SphericalKernel(FloatType sigma): sigma_inv_(1 / sigma), sigma_(sigma) {}
    std::string str() const {
        return std::string("SphericalKernel:{") + std::to_string(sigma_) + '}';
    }
};

template<typename FloatType>
struct GeneralHistogramKernel: KernelBase<FloatType> {
    const FloatType a_;
    const FloatType b_;
    template<typename MatrixType1, typename MatrixType2>
    FloatType operator()(const MatrixType1 &a, const MatrixType2 &b) const {
        if(a.size() != b.size())
            throw std::out_of_range(std::string("Could not calculate min between arrays of different sizes (") + std::to_string(a.size()) + ", " + std::to_string(b.size()) + ")");
        auto ait(a.cbegin()), bit(b.cbegin());
        FloatType ret(std::min(*ait, *bit));
        while(++ait != a.cend()) ret += std::min(std::abs(std::pow(*ait, a_)), std::abs(std::pow(*++bit, b_)));
        return ret;
    }
    GeneralHistogramKernel(FloatType a, FloatType b): a_(a), b_(b) {}
    std::string str() const {
        return std::string("GeneralHistogramKernel:{") +
            std::to_string(a_) + ", " + std::to_string(b_) + '}';
    }
};

template<typename FloatType>
struct CauchyKernel: KernelBase<FloatType> {
    const FloatType sigma_sq_inv_;
    template<typename MatrixType1, typename MatrixType2>
    FloatType operator()(const MatrixType1 &a, const MatrixType2 &b) const {
        return 1. / (1. + diffnorm(a, b) * sigma_sq_inv_);
    }
    CauchyKernel(FloatType sigma): sigma_sq_inv_(1. / (sigma * sigma)) {}
    std::string str() const {
        return std::string("CauchyKernel:{") +
            std::to_string(std::sqrt(1./sigma_sq_inv_)) + '}';
    }
};

template<typename FloatType>
struct ChiSqPDVariantKernel: KernelBase<FloatType> {
    template<typename MatrixType1, typename MatrixType2>
    FloatType operator()(const MatrixType1 &a, const MatrixType2 &b) const {
        return 2. * dot(a, b / (a + b));
    }
    std::string str() const {
        return "ChiSqPDVariantKernel";
    }
};


template<typename FloatType>
struct ChiSqKernel: KernelBase<FloatType> {
    template<typename MatrixType1, typename MatrixType2>
    FloatType operator()(const MatrixType1 &a, const MatrixType2 &b) const {
        return 1. - 2. * dot(a - b, ((a - b) / (a + b)));
    }
    std::string str() const {
        return "ChiSqKernel";
    }
};

template<typename FloatType>
struct StudentKernel: KernelBase<FloatType> {
    const FloatType d_;
    template<typename MatrixType1, typename MatrixType2>
    FloatType operator()(const MatrixType1 &a, const MatrixType2 &b) const {
        return 1. / (1 + std::pow(diffnorm(a, b), d_));
    }
    StudentKernel(FloatType d): d_(d / 2.) {} // Divide by 2 to get the n-nom.
    std::string str() const {
        return std::string("StudentKernel:{") + std::to_string(d_) + '}';
    }
};

template<typename FloatType>
struct ANOVAKernel: KernelBase<FloatType> {
    const FloatType d_, k_, sigma_;
    template<typename MatrixType1, typename MatrixType2>
    FloatType operator()(const MatrixType1 &a, const MatrixType2 &b) const {
        return std::pow(
            -sigma_ * diffnorm(blaze::pow(a, k_), blaze::pow(b, k_)), d_);
    }
    ANOVAKernel(FloatType d, FloatType k, FloatType sigma): d_(d / 2.), k_(k), sigma_(sigma) {}
    std::string str() const {
        return std::string("ANOVAKernel:{") + std::to_string(d_ * 2.) +
            ", " + std::to_string(k_) + ", " + std::to_string(sigma_) + '}';
    }
};

template<typename FloatType>
struct DefaultWaveletFunction {
    FloatType operator()(FloatType input) const {
        return std::cos(1.75 * input) * std::exp(input * input * -0.5);
    }
};

template<typename FloatType, class WaveletFunction=DefaultWaveletFunction<FloatType>>
struct WaveletKernel: KernelBase<FloatType> {
    const FloatType a_inv_, c_;
    WaveletFunction fn_;
    template<typename MatrixType1, typename MatrixType2>
    FloatType operator()(const MatrixType1 &a, const MatrixType2 &b) const {
        auto ait(a.cbegin()), bit(b.cbegin());
        FloatType ret(fn_((*ait - c_) * a_inv_) * fn_((*bit - c_) * a_inv_));
        while(++ait != a.cend() && ++bit != b.cend()) ret *= fn_((*ait - c_) * a_inv_) * fn_((*bit - c_) * a_inv_);
        return ret;
    }
    WaveletKernel(FloatType a, FloatType c): a_inv_(1. / a), c_(c) {}
    std::string str() const {
        return std::string("WaveletKernel:{") + std::to_string(1. / a_inv_) +
            ", " + std::to_string(c_) + '}';
    }
};

template<typename FloatType>
struct LogarithmicKernel: KernelBase<FloatType> {
    const FloatType d_;
    template<typename MatrixType1, typename MatrixType2>
    FloatType operator()(const MatrixType1 &a, const MatrixType2 &b) const {
        return -std::log(std::pow(diffnorm(a, b), d_) + 1);
    }
    LogarithmicKernel(FloatType d): d_(d / 2.) {} // Divide by 2 to get the n-norm.
    std::string str() const {
        return std::string("LogarithmicKernel:{") + std::to_string(d_) + '}';
    }
};

template<typename FloatType>
struct HistogramKernel: KernelBase<FloatType> {
    template<typename MatrixType1, typename MatrixType2>
    FloatType operator()(const MatrixType1 &a, const MatrixType2 &b) const {
        if(a.size() != b.size())
            throw std::out_of_range(std::string(
                "Could not calculate min for arrays of different sizes (") +
                std::to_string(a.size()) + ", " +
                std::to_string(b.size()) + ')');
        auto ait(a.cbegin()), bit(b.cbegin());
        FloatType ret(std::min(*ait, *bit));
        while(++ait != a.cend()) ret += std::min(*ait, *++bit);
        return ret;
    }
    std::string str() const {
        return "HistogramKernel";
    }
};


template<typename FloatType>
struct ExponentialBesselKernel: KernelBase<FloatType> {
    // https://github.com/primaryobjects/Accord.NET/blob/master/Sources/Accord.Statistics/Kernels/Bessel.cs
    // https://github.com/DiegoCatalano/Catalano-Framework/blob/master/Catalano.Statistics/src/Catalano/Statistics/Kernels/Bessel.java
    const FloatType sigma_, order_;
    template<typename MatrixType1, typename MatrixType2>
    FloatType operator()(const MatrixType1 &a, const MatrixType2 &b) const {
        const auto norm(std::sqrt(diffnorm(a, b)));
        return cyl_bessel_j(order_, sigma_ * norm) /
            std::exp(norm, -norm * order_);
    }
    ExponentialBesselKernel(FloatType sigma, FloatType order):
        sigma_(sigma), order_(order) {}
    std::string str() const {
        return std::string("ExponentialBesselKernel:{") +
            std::to_string(sigma_) +
            ", " + std::to_string(order_) + '}';
    }
};


template<typename FloatType>
struct CylindricalBesselKernel: KernelBase<FloatType> {
    // http://crsouza.com/2010/03/17/kernel-functions-for-machine-learning-applications/#bessel
    const FloatType sigma_, vp1_, minus_nvp1_;
    template<typename MatrixType1, typename MatrixType2>
    FloatType operator()(const MatrixType1 &a, const MatrixType2 &b) const {
        const auto norm(sigma_ * std::sqrt(diffnorm(a, b)));
        return cyl_bessel_j(vp1_, sigma_ * norm) / std::exp(norm, minus_nvp1_);
    }
    CylindricalBesselKernel(FloatType sigma, FloatType n, FloatType v):
        sigma_(sigma), vp1_(v + 1), minus_nvp1_(-n * vp1_) {}
    std::string str() const {
        return std::string("CylindricalBesselKernel:{") +
            std::to_string(sigma_) + ", " +
            std::to_string(minus_nvp1_ / -(vp1_ - 1)) + ", " +
            std::to_string(vp1_ - 1) + '}';
    }
};

template<typename FloatType>
struct CircularKernel: KernelBase<FloatType> {
    const FloatType sigma_inv_;
    const FloatType sigma_;
    static constexpr FloatType TWO_OVER_PI = 2 / M_PIl;
    template<typename MatrixType1, typename MatrixType2>
    FloatType operator()(const MatrixType1 &a, const MatrixType2 &b) const {
        const FloatType norm2(std::sqrt(diffnorm(a, b)) * sigma_inv_);
        return norm2 >= 1. ? 0.
                           : TWO_OVER_PI * (std::acos(-norm2) - norm2 *
                                            std::sqrt(1 - norm2 * norm2));
    }
    CircularKernel(FloatType sigma): sigma_inv_(1 / sigma), sigma_(sigma) {}
    std::string str() const {
        return std::string("CircularKernel:{") +
            std::to_string(sigma_) + '}';
    }
};

template<typename FloatType>
struct TanhKernel: KernelBase<FloatType>{
    const FloatType k_;
    const FloatType c_;
    template<typename MatrixType1, typename MatrixType2>
    FloatType operator()(const MatrixType1 &a, const MatrixType2 &b) const {
        return std::tanh(dot(a, b) * k_ + c_);
    }
    TanhKernel(FloatType k, FloatType c): k_(k), c_(c) {}
    std::string str() const {
        return std::string("TanhKernel:{") +
            std::to_string(k_) + ", " + std::to_string(c_) + '}';
    }
};

} // namespace svm

#endif // _KERNEL_H_
