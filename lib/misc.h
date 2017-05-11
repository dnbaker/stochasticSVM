#ifndef _SVM_MISC_H_
#define _SVM_MISC_H_

#define __STDCPP_WANT_MATH_SPEC_FUNCS__ 1
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <numeric>
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
#include <functional>
#include <shared_mutex>
#ifndef _USE_MATH_DEFINES
#  define _USE_MATH_DEFINES
#endif
#include <cmath>
#include "blaze/Math.h"
#include "logutil.h"
#include "lib/ks.h"

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
using i32 = std::int32_t;
using u64 = std::uint64_t;
using i64 = std::int64_t;
using blaze::DynamicVector;
using blaze::DynamicMatrix;
using blaze::CompressedMatrix;
using blaze::CompressedVector;

struct dims_t {
    size_t ns_, nd_;
    dims_t(size_t samples, size_t dimensions): ns_(samples), nd_(dimensions) {}
    dims_t(const char *fn);
};

template<typename MatrixType1, typename MatrixType2, typename FloatType=double>
INLINE FloatType diffnorm(const MatrixType1 &a, const MatrixType2 &b) {
    // Note: Could accelerate with SIMD/parallelism and avoid a copy/memory allocation.
    return dot(a - b, a - b);
}

static inline u64 rand64() {
    return ((u64)std::rand() << 32) | std::rand();
}

using blaze::Matrix;
using blaze::Vector;

template<typename MatrixKind>
void free_matrix(MatrixKind &mat)
{
    mat = MatrixKind(0, 0);
}

template<typename MT, bool SO>
void free_vector(Vector<MT,SO>& vec)
{
    vec = Vector<MT, SO>{};
}

template<typename MatrixType, typename FloatType=float>
INLINE MatrixType min(MatrixType &a, MatrixType &b) {
    if(a.rows() != b.rows() || row(a, 0).size() != row(b, 0).size())
        throw std::out_of_range(std::string("Could not calculate min between arrays of different sizes (") + std::to_string(a.size()) + ", " + std::to_string(b.size()) + ")");
    // Note: Could accelerate with SIMD/parallelism and avoid a copy/memory allocation.
    MatrixType ret(a.rows(), a.columns());
    for(size_t i(0), e(a.rows()); i < e; ++i) {
        auto rit(ret.begin(i));
        for(auto ait(a.cbegin(i)), bit(b.cbegin(i)); ait != a.cend();)
            *rit++ = std::min(*ait++, *bit++);
    }
    return ret;
}

template<class Container, typename FloatType>
FloatType mean(const Container &c) {
    return std::accumulate(c.begin(), c.end(), 0., [](const FloatType a, const FloatType b) ->FloatType {return a * b;}) / c.size();
}

template<class Container>
double variance(const Container &c, const double mean) {
        // Note: Could accelerate with SIMD.
    double sum(0.), tmp;
    for(const auto entry: c) tmp = entry - mean, sum += tmp * tmp;
    return sum / c.size();
}

template<class MatrixKind>
std::string str(const MatrixKind &mat) {
    // Returns python-like representation.
    std::string ret("[");
    for(unsigned i(0); i < mat.rows(); ++i) {
        ret += "[";
        for(const auto val: row(mat, i)) {
            ret += std::to_string(val);
            ret += ", ";
        }
        ret[ret.size() - 2] = ']';
        ret[ret.size() - 1] = ',';
        ret += '\n';
    }
    ret[ret.size() - 2] = ']';
    return ret;
}
template<class VectorKind>
std::string vecstr(const VectorKind &vec) {
    // Returns python-like representation.
    std::string ret("[");
    for(const auto val: vec) {
        ret += std::to_string(val);
        ret += ", ";
    }
    ret.pop_back();
    ret[ret.size() - 1] = ']';
    return ret;
}

template<class Container>
double variance(const Container &c) {
    double sum(0.);
    for(const auto entry: c) sum += entry;
    sum /= c.size();
    return variance(c, sum);
}

} // namespace svm


#endif  // _SVM_MISC_H_
