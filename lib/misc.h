#ifndef _SVM_MISC_H_
#define _SVM_MISC_H_

#define __STDCPP_WANT_MATH_SPEC_FUNCS__ 1
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
#include <stdexcept>
#ifndef _USE_MATH_DEFINES
#  define _USE_MATH_DEFINES
#endif
#ifdef __GNUC__
#  if (__GNUC__ == 6 && __GNUC_MINOR__ >= 1) || __GNUC__ > 6
#    include <cmath>
#  else
#    include <tr1/cmath>
#  endif
#else
#  include <cmath>
#endif
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

#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif

#ifndef M_PIl
#define M_PIl (3.14159265358979323846264338327950288)
#endif

#if defined(__STDCPP_MATH_SPEC_FUNCS__) || _GLIBCXX_TR1_CMATH
  #if __STDCPP_MATH_SPEC_FUNCS__ >= 201003L
    using std::cyl_bessel_j;
  #else
  #define STRINGIFY(s) XSTRINGIFY(s)
  #define XSTRINGIFY(s) #s
  #pragma message "Getting bessel function from trl with __STDCPP_WANT_MATH_SPEC_FUNCS__ as " STRINGIFY(__STDCPP_WANT_MATH_SPEC_FUNCS__)
  #undef STRINGIFY
  #undef XSTRINGIFY
    using std::tr1::cyl_bessel_j;
  #endif
#else
#pragma message "Getting bessel function from boost"
#include "boost/math/special_functions/bessel.hpp"
using boost::math::cyl_bessel_j;
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
    return dot(a - b, a - b);
}

template<typename MatrixType, typename FloatType=float>
INLINE MatrixType min(MatrixType &a, MatrixType &b) {
    if(a.size() != b.size())
        throw std::out_of_range(std::string("Could not calculate min between arrays of different sizes (") + std::to_string(a.size()) + ", " + std::to_string(b.size()) + ")");
    // Note: Could accelerate with SIMD/parallelism and avoid a copy/memory allocation.
    DynamicVector<FloatType> ret(a.size());
    auto rit(ret.begin());
    for(auto ait(a.cbegin()), bit(b.cbegin()); ait != a.cend(); ++ait, ++bit)
        *rit++ = std::min(*ait++, *bit++);
    return ret;
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
