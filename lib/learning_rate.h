#ifndef _LEARNING_RATE_H
#define _LEARNING_RATE_H

namespace svm {

template<typename FloatType>
struct PegasosLearningRate {
    FloatType lambda_inv_;
    FloatType operator()(u64 t) const {
        return lambda_inv_ / (t + 1);
    }
    PegasosLearningRate(FloatType lambda): lambda_inv_(1./lambda) {}
    PegasosLearningRate(): lambda_inv_(1.) {}
};

template<typename FloatType>
struct FixedLearningRate {
    FloatType eta_;
    FloatType operator()(u64 t) const {
        return eta_;
    }
    FixedLearningRate(FloatType eta): eta_(eta) {}
    FixedLearningRate(): eta_(1.) {}
};

template<typename FloatType>
struct NormaLearningRate {
    FloatType eta_;
    FloatType operator()(u64 t) const {
        return eta_ / std::sqrt(t + 1);
    }
    NormaLearningRate(FloatType eta): eta_(eta) {}
    NormaLearningRate(): eta_(1.) {}
};


} // namespace svm

#endif
