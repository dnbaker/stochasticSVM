#ifndef _SVM_MISC_H_
#define _SVM_MISC_H_

#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cassert>
#include <thread>
#include <map>
#include <unordered_map>
#include <set>
#include <limits>
#include <zlib.h>
#include <iostream>
#include <fstream>
#include "blaze/Math.h"
#include "logutil.h"
#include "klib/kstring.h"


#ifdef __GNUC__
#  define likely(x) __builtin_expect((x),1)
#  define unlikely(x) __builtin_expect((x),0)
#  define UNUSED(x) __attribute__((unused)) x
#else
#  define likely(x) (x)
#  define unlikely(x) (x)
#  define UNUSED(x) (x)
#endif

#ifndef INLINE
#  if __GNUC__ || __clang__
#  define INLINE __attribute__((always_inline)) inline
#  else
#  define INLINE inline
#  endif
#endif

namespace svm {

using std::cerr;
using std::cout;

using std::size_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using blaze::DynamicVector;
using blaze::DynamicMatrix;

template<typename FloatType, typename MatrixType1, typename MatrixType2>
FloatType dot(MatrixType1 &a, MatrixType2 &b) {
    return static_cast<FloatType>(a * trans(b));
}

struct dims_t {
    size_t ns_, nd_;
    dims_t(size_t samples, size_t dimensions): ns_(samples), nd_(dimensions) {}
    dims_t(const char *fn);
};

template<typename MatrixType, typename FloatType=float>
INLINE FloatType diffnorm(MatrixType &a, MatrixType &b) {
    // Note: Could accelerate with SIMD/parallelism and avoid a copy/memory allocation.
    const auto diff(a - b);
    return dot(diff, diff);
}

template<class Container, typename FloatType>
FloatType variance(const Container &c, const FloatType mean) {
        // Note: Could accelerate with SIMD.
    FloatType sum(0.), tmp;
    for(auto entry: c) tmp = entry - mean, sum += tmp * tmp;
    return sum / c.size();
}
template<class Container, typename FloatType>
FloatType variance(const Container &c) {
    FloatType sum(0.);
    for(auto entry: c) sum += c;
    return variance(c, sum / c.size());
}

} // namespace svm


#endif  // _SVM_MISC_H_
