#ifndef _LEARNING_RATE_H
#define _LEARNING_RATE_H

namespace svm {

template<typename FloatType>
struct PegasosLearningRate {
    const FloatType lambda_inv_;
    FloatType operator()(u64 t) {
        return lambda_inv_ / (t + 1);
    }
    PegasosLearningRate(FloatType lambda): lambda_inv_(1./lambda) {}
};

template<typename FloatType>
struct FixedLearningRate {
    const FloatType eta_;
    FloatType operator()(u64 t) {
        return eta_;
    }
    FixedLearningRate(FloatType eta): eta_(eta) {}
};

template<typename FloatType>
struct NormaLearningRate {
    const FloatType eta_;
    FloatType operator()(u64 t) {
        return eta_ / std::sqrt(t + 1);
    }
    NormaLearningRate(FloatType eta): eta_(eta) {}
};


} // namespace svm

#endif
