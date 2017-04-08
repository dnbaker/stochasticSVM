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

namespace svm {

using std::size_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using blaze::DynamicVector;
using blaze::DynamicMatrix;

std::pair<size_t, unsigned> count_dims(const char *fn, size_t bufsize=1<<16);

} // namespace svm


#endif  // _SVM_MISC_H_
