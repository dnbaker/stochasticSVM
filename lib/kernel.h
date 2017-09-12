#ifndef _KERNEL_H_
#define _KERNEL_H_
#include "lib/rand.h"
#include "lib/misc.h"
#include "blaze/Math.h"

namespace svm {

template<typename FloatType>
struct KernelBase {
    using float_type = FloatType;
    template<typename RowType1, typename RowType2>
    INLINE FloatType operator()(const RowType1 &a, const RowType2 &b) const;
    template<typename RowType>
    INLINE void sample(RowType &row) const {
        throw NotImplementedError("Sampling must be overridden.");
    }
    template<typename RowType1, typename RowType2, typename TmpRowType, typename PRNGen=rng::RandTwister>
    INLINE void rff_sample(const RowType1 &in, RowType2 &out, size_t d, TmpRowType &tmp, const PRNGen &rng=PRNGen{}) const {
        // samples from Gaussian distribution randomly for RFF generation.
        // in:  blaze::RowType of some kind, unmodifid.
        // out: blaze::RowType of some kind, set.
        // d:   size of output data.
        if(unlikely(out.size() != d)) {
            throw std::runtime_error(("Output array of length "s +
                                      std::to_string(out.size()) +
                                      " does not match expected/required " +
                                      std::to_string(d)).data());
        }
        assert((d & 1) == 0); // Must be even
        throw NotImplementedError("No rff_sample method provided, but only because I haven't figured it out yet!\n");
        if(tmp.size() != in.size()) {
            std::cerr << "Resizing temporary array from " << tmp.size() << " to  " << in.size();
            tmp.resize(in.size());
        }
        // Consider caching an array of 0,PI/2,0,PI/2 and adding performing ::blaze::sin(pi_arr + <filled_random>)
        // It might be faster (SIMD), it might be slower (two passes, more to compete for cache).
        for(size_t i(0); i < d;) {
            // TODO: manually SIMD the sin calls using SIMD and template longer loop unrolling.
            this->sample(tmp);
            out[i++] = std::sin(dot(tmp, in));
            this->sample(tmp);
            out[i++] = std::cos(dot(tmp, in));
        }
    }
    std::string str() const {
        throw NotImplementedError("No .std() method provided.");
        return "KernelBase";
    }
    // TODO: Add SimdEnabled methods to these kernels for SIMD parallelism.
};

template<typename FloatType>
struct LinearKernel: KernelBase<FloatType> {
    template<typename RowType1, typename RowType2>
    INLINE FloatType operator()(const RowType1 &a, const RowType2 &b) const {
        return dot(a, b);
    }
    template<typename RowType>
    INLINE void sample(RowType &row) const {
        throw NotImplementedError("No rff_sample method provided for linear kereel as it is redundant. [Won't Fix]\n");
    }
    LinearKernel() {}
    std::string str() const {return "LinearKernel";}
};

template<typename FloatType>
struct PolyKernel: KernelBase<FloatType> {
    const FloatType a_;
    const FloatType c_;
    const FloatType d_;
    template<typename RowType1, typename RowType2>
    INLINE FloatType operator()(const RowType1 &a, const RowType2 &b) const {
        return std::pow(a_ * dot(a, b) + c_, d_);
    }
    PolyKernel(FloatType a, FloatType c, FloatType d): a_(a), c_(c), d_(d) {}
    std::string str() const {
        return std::string("PolyKernel:{") + std::to_string(a_) + ", " +
                           std::to_string(c_) + ", " +
                           std::to_string(d_) + "}";
    }
};

template<typename FloatType>
struct LaplacianKernel: KernelBase<FloatType> {
    const FloatType msigma_;
    template<typename RowType1, typename RowType2>
    INLINE FloatType operator()(const RowType1 &a, const RowType2 &b) const {
        return std::exp(msigma_ * std::sqrt(
            diffnorm<RowType1, RowType2, FloatType>(a, b)));
    }
    LaplacianKernel(FloatType sigma): msigma_(-sigma) {}
    std::string str() const {
        return std::string("LaplacianKernel:{") +
            std::to_string(-msigma_) + '}';
    }
};

template<typename FloatType>
struct RationalQuadKernel: KernelBase<FloatType> {
    const FloatType c_;
    template<typename RowType1, typename RowType2>
    INLINE FloatType operator()(const RowType1 &a, const RowType2 &b) const {
        const FloatType dn(diffnorm(a, b));
        return 1. - (dn / dn + c_);
    }
    RationalQuadKernel(FloatType c): c_(c) {}
    std::string str() const {
        return std::string("RationalQuadKernel:{") +
            std::to_string(c_) + '}';
    }
};

template<typename FloatType>
struct MultiQuadKernel: KernelBase<FloatType> {
    const FloatType c2_;
    template<typename RowType1, typename RowType2>
    INLINE FloatType operator()(const RowType1 &a, const RowType2 &b) const {
        return std::sqrt(diffnorm(a, b) + c2_);
    }
    MultiQuadKernel(FloatType c): c2_(c * c) {}
    std::string str() const {
        return std::string("MultiQuadKernel:{") +
            std::to_string(std::sqrt(c2_)) + '}';
    }
};

template<typename FloatType>
struct InvMultiQuadKernel: KernelBase<FloatType> {
    const FloatType c2_;
    template<typename RowType1, typename RowType2>
    INLINE FloatType operator()(const RowType1 &a, const RowType2 &b) const {
        return 1. / (std::sqrt(diffnorm(a, b) + c2_));
    }
    InvMultiQuadKernel(FloatType c): c2_(c * c) {}
    std::string str() const {
        return std::string("InvMultiQuadKernel:{") +
            std::to_string(std::sqrt(c2_)) + '}';
    }
};

template<typename FloatType>
struct GaussianKernel: KernelBase<FloatType> {
    const FloatType mgamma_;
    GaussianKernel(FloatType gamma): mgamma_(-gamma * 0.5) {}

    template<typename RowType1, typename RowType2>
    INLINE FloatType operator()(const RowType1 &a, const RowType2 &b) const {
        return std::exp(mgamma_ *
                        diffnorm<RowType1, RowType2, FloatType>(a, b));
    }
    template<typename RowType>
    INLINE void sample(RowType &row) {
        // TODO: *this*
        std::cerr << "Doing nothing in this sampling.";
    }
    std::string str() const {
        return std::string("GaussianKernel:{") + std::to_string(-mgamma_) + '}';
    }
};

template<typename FloatType>
struct ExpKernel: KernelBase<FloatType> {
    const FloatType mgamma_;
    template<typename RowType1, typename RowType2>
    INLINE FloatType operator()(const RowType1 &a, const RowType2 &b) const {
        return std::exp(mgamma_ * std::sqrt(
            diffnorm<RowType1, RowType2, FloatType>(a, b)));
    }
    ExpKernel(FloatType gamma): mgamma_(-gamma) {}
    std::string str() const {
        return std::string("ExpKernel:{") + std::to_string(-mgamma_) + '}';
    }
};

template<typename FloatType>
struct SphericalKernel: KernelBase<FloatType> {
    const FloatType sigma_inv_;
    const FloatType sigma_;
    template<typename RowType1, typename RowType2>
    INLINE FloatType operator()(const RowType1 &a, const RowType2 &b) const {
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
    template<typename RowType1, typename RowType2>
    INLINE FloatType operator()(const RowType1 &a, const RowType2 &b) const {
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
    template<typename RowType1, typename RowType2>
    INLINE FloatType operator()(const RowType1 &a, const RowType2 &b) const {
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
    template<typename RowType1, typename RowType2>
    INLINE FloatType operator()(const RowType1 &a, const RowType2 &b) const {
        return 2. * dot(a, b / (a + b));
    }
    std::string str() const {
        return "ChiSqPDVariantKernel";
    }
};


template<typename FloatType>
struct ChiSqKernel: KernelBase<FloatType> {
    template<typename RowType1, typename RowType2>
    INLINE FloatType operator()(const RowType1 &a, const RowType2 &b) const {
        return 1. - 2. * dot(a - b, ((a - b) / (a + b)));
    }
    std::string str() const {
        return "ChiSqKernel";
    }
};

template<typename FloatType>
struct StudentKernel: KernelBase<FloatType> {
    const FloatType d_;
    template<typename RowType1, typename RowType2>
    INLINE FloatType operator()(const RowType1 &a, const RowType2 &b) const {
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
    template<typename RowType1, typename RowType2>
    INLINE FloatType operator()(const RowType1 &a, const RowType2 &b) const {
        return std::pow(
            -sigma_ * diffnorm(blaze::pow(a, k_), blaze::pow(b, k_)), d_);
    }
    ANOVAKernel(FloatType d, FloatType k, FloatType sigma): d_(d / 2.), k_(k), sigma_(sigma) {}
    std::string str() const {
        return std::string("ANOVAKernel:{") + std::to_string(d_ * 2.) +
            ", " + std::to_string(k_) + ", " + std::to_string(sigma_) + '}';
    }
};

template<size_t degree>
struct ArccosKernelJDetail {
    double operator()(double theta) const {
        throw std::runtime_error("NotImplementedError");
        return 0.;
    }
};

template<> struct ArccosKernelJDetail<0> {
    double operator()(double theta) const {return M_PI - theta;}
};

template<>
struct ArccosKernelJDetail<1> {
    double operator()(double theta) const {return std::sin(theta) + (M_PI - theta) * std::cos(theta);}
};

template<>
struct ArccosKernelJDetail<2> {
    double operator()(double theta) const {
        const double c(std::cos(theta));
        return 3. * std::sin(theta) * c + (M_PI - theta) * (1 + 2 * c * c);
    }
};

template<typename FloatType, size_t degree=0>
struct ArccosKernel: KernelBase<FloatType> {
    // https://papers.nips.cc/paper/3628-kernel-methods-for-deep-learning.pdf
    ArccosKernelJDetail<degree> fn_;
    template<typename RowType1, typename RowType2>
    INLINE FloatType operator()(const RowType1 &a, const RowType2 &b) const {
        const FloatType anorm(svm::norm(a)), bnorm(svm::norm(b)), theta(std::acos(dot(a, b) / (anorm * bnorm)));
        return std::pow(anorm * bnorm, degree) / M_PI * fn_(theta);
    }
    std::string str() const {
        return std::string("ArccosKernel<") + std::to_string(degree) + ">";
    }
};

template<typename FloatType>
struct DefaultWaveletFunction {
    INLINE FloatType operator()(FloatType input) const {
        return std::cos(1.75 * input) * std::exp(input * input * -0.5);
    }
};

template<typename FloatType, class WaveletFunction=DefaultWaveletFunction<FloatType>>
struct WaveletKernel: KernelBase<FloatType> {
    const FloatType a_inv_, c_;
    WaveletFunction fn_;
    template<typename RowType1, typename RowType2>
    INLINE FloatType operator()(const RowType1 &a, const RowType2 &b) const {
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
    template<typename RowType1, typename RowType2>
    INLINE FloatType operator()(const RowType1 &a, const RowType2 &b) const {
        return -std::log(std::pow(diffnorm(a, b), d_) + 1);
    }
    LogarithmicKernel(FloatType d): d_(d / 2.) {} // Divide by 2 to get the n-norm.
    std::string str() const {
        return std::string("LogarithmicKernel:{") + std::to_string(d_) + '}';
    }
};

template<typename FloatType>
struct HistogramKernel: KernelBase<FloatType> {
    template<typename RowType1, typename RowType2>
    INLINE FloatType operator()(const RowType1 &a, const RowType2 &b) const {
        FloatType ret(0.);
        if constexpr(!blaze::IsSparseVector<RowType1>::value && !blaze::IsSparseVector<RowType2>::value) {
            for(auto ait(a.cbegin()), bit(b.cbegin()), eait(a.cend()); ait != eait; ++ait, ++bit) {
                ret += std::min(*ait, *bit);
            }
        } else if constexpr(blaze::IsSparseVector<RowType1>::value && !blaze::IsSparseVector<RowType2>::value) {
            for(auto ait(a.cbegin()), eait(a.cend()); ait != eait; ++ait) {
                ret += std::min(a->value(), b[a->index()]);
            }
        } else if constexpr(!blaze::IsSparseVector<RowType1>::value && blaze::IsSparseVector<RowType2>::value) {
            for(auto ait(b.cbegin()), eait(b.cend()); ait != eait; ++ait) {
                ret += std::min(b->value(), a[b->index()]);
            }
        } else { // blaze::IsSparseVector<RowType1>::value && blaze::IsSparseVector<RowType2>::value)
            auto ait(a.cbegin()), eait(a.cend()), bit(b.cbegin()), ebit(b.cend());
            while(ait != eait && bit != ebit) {
                if(ait->index() < bit->index()) {
                    ++ait;
                    continue;
                } else if(ait->index() > bit->index()) {
                    ++bit;
                    continue;
                }
                ret += std::min(ait->value(), bit->value());
            }
        }
        return ret;
#if 0
        if(a.size() != b.size())
            throw std::out_of_range(std::string(
                "Could not calculate min for arrays of different sizes (") +
                std::to_string(a.size()) + ", " +
                std::to_string(b.size()) + ')');
        auto ait(a.cbegin()), bit(b.cbegin());
        FloatType ret(std::min(*ait, *bit));
        while(++ait != a.cend()) ret += std::min(*ait, *++bit);
        return ret;
#endif
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
    template<typename RowType1, typename RowType2>
    INLINE FloatType operator()(const RowType1 &a, const RowType2 &b) const {
        const auto norm(std::sqrt(diffnorm(a, b)));
        return cyl_bessel_j(order_, sigma_ * norm) /
            std::pow(norm, -norm * order_);
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
    template<typename RowType1, typename RowType2>
    INLINE FloatType operator()(const RowType1 &a, const RowType2 &b) const {
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
    template<typename RowType1, typename RowType2>
    INLINE FloatType operator()(const RowType1 &a, const RowType2 &b) const {
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
    template<typename RowType1, typename RowType2>
    INLINE FloatType operator()(const RowType1 &a, const RowType2 &b) const {
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
