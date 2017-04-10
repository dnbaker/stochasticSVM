#ifndef _LEARNING_RATE_H
#define _LEARNING_RATE_H

namespace svm {

template<typename FloatType>
struct PegasosLearningRate {
    const FloatType lambda_inv_;
    FloatType operator()(u64 t) {
        return lambda_inv_ / t;
    }
    PegasosLearningRate(FloatType lambda): lambda_inv_(1./lambda) {}
};

} // namespace svm

#endif
