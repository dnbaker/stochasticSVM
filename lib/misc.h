#ifndef _SVM_MISC_H_
#define _SVM_MISC_H_

#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cassert>
#include "blaze/Math.h"
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

using std::size_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using blaze::DynamicVector;
using blaze::DynamicMatrix;

std::pair<size_t, unsigned> count_dims(const char *fn, size_t bufsize=1<<16);

template<typename MatrixType, typename FloatType=float>
INLINE dot(MatrixType &a, MatrixType &b) {
    return a * trans(b);
}

template<typename MatrixType, typename FloatType=float>
INLINE diffnorm(MatrixType &a, MatrixType &b) {
    const auto norm(a - b);
    return dot(norm, norm);
}

} // namespace svm


#endif  // _SVM_MISC_H_
