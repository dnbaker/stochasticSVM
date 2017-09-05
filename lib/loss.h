#ifndef _LOSS_H__
#define _LOSS_H__
#include "lib/misc.h"

namespace svm {

template<typename FloatType>
struct HingeSubgradientCore {
    template<typename SVMType, typename TmpRowType>
    INLINE void operator()(SVMType &svm, TmpRowType &trow,
                           khash_t(I) *h) const {
        const auto &mat = svm.m();
        const auto &v(svm.v());
        for(size_t i(0); i < kh_size(h); ++i) {
            if(!kh_exist(h, i)) continue;
            const size_t index(kh_key(h, i));
            if(svm.predict(row(mat, index)) * v[index] < 1.) {
                trow += row(mat, index) * v[index];
            }
        }
    }
    HingeSubgradientCore() {}
};

template<typename FloatType>
struct LogisticSubgradientCore {
    template<typename SVMType, typename TmpRowType>
    INLINE void operator()(SVMType &svm, TmpRowType &trow,
                           khash_t(I) *h) const {
        const auto &mat = svm.m();
        const auto &v(svm.v());
        for(size_t i(0); i < kh_size(h); ++i) {
            if(!kh_exist(h, i)) continue;
            const size_t index(kh_key(h, i));
            trow += row(mat, index) * (v[index] / (1 + std::exp(svm.kernel(row(mat, index), row(svm.w().weights(), 0)) * v[index])));
        }
    }
    LogisticSubgradientCore() {}
};

template<typename FloatType, typename SubgradientCore=HingeSubgradientCore<FloatType>>
struct LossSubgradient {
    const SubgradientCore core_;

    template<typename SVMType, typename TmpRowType, typename LastWeightsType>
    INLINE void operator() (SVMType &svm, TmpRowType &trow,
                            LastWeightsType &last_weights, khash_t(I) *h) const {
        const auto eta(svm.lp()(svm.t()));
        trow = 0.;
        core_(svm, trow, h);
        if(svm.eps() > 0 && svm.t() < svm.max_iter()) last_weights = svm.w().weights_;
        svm.w().scale(1.0 - svm.lambda() * eta);
        row(svm.w().weights_, 0) += trow * (eta / static_cast<FloatType>(kh_size(h)));
    }
    LossSubgradient(): core_{} {}
};

} // namespace svm

#endif // #ifndef _LOSS_H__
