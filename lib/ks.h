#ifndef _KS_WRAPPER_H__
#define _KS_WRAPPER_H__

#include <cstdint>
#include <cstdio>
#include <iostream>
#include <unistd.h>
#include "klib/kstring.h"

namespace ks {

class KString {
    kstring_t ks_;

public:

    explicit KString(size_t size): ks_({0, size, size ? (char *)std::malloc(size): nullptr}) {}
    explicit KString(size_t used, size_t max, char *str): ks_({used, max, str}) {}
    explicit KString(char *str): ks_({0, 0, str}) {}

    KString(): KString(nullptr) {}
    ~KString() {free(ks_.s);}

    // kstring_t access:
    const auto operator->() const {
        return const_cast<const kstring_t *>(&ks_);
    }
    kstring_t *operator->() {return &ks_;}
    const auto &operator*() const {
        return const_cast<const kstring_t &>(ks_);
    }
    kstring_t  &operator*() {return ks_;}

    // Conversions
    operator const char *() const {return ks_.s;}
    operator       char *()       {return ks_.s;}

    operator const kstring_t *() const {return &ks_;}
    operator       kstring_t *()       {return &ks_;}
    // Copy
    KString(const KString &other): ks_{other->l, other->m, (char *)std::malloc(other->m)} {
        memcpy(ks_.s, other->s, other->m);
    }

    // Move
    KString(KString &&other) {
        memcpy(this, &other, sizeof(other));
        memset(&other, 0, sizeof(other));
    }

    // Comparison functions
    int cmp(const char *s) {
        return strcmp(ks_.s, s);
    }
    int cmp(const KString &other) {return cmp(other->s);}

    bool operator==(const KString &other) {
        if(other->l != ks_.l) return 0;
        for(size_t i(0); i < ks_.l; ++i) if(ks_.s[i] != other->s[i]) return 0;
        return 1;
    }

    bool palindrome() {
        for(size_t i(0), e(ks_.l >> 1); i < e; ++i)
            if(ks_.s[i] != ks_.s[ks_.l - i - 1])
                return 0;
        return 1;
    }

    // Appending:
    int putc(int c) {return kputc(c, &ks_);}
    int putc_(int c) {return kputc_(c, &ks_);}
    int putw(int c) {return kputw(c, &ks_);}
    int putl(int c) {return kputl(c, &ks_);}
    int putuw(int c) {return kputuw(c, &ks_);}
    int puts(const char *s) {return kputs(s, &ks_);}
    int putsn(const char *s, int l) {return kputsn(s, l, &ks_);}
    int putsn_(const char *s, int l) {return kputsn_(s, l, &ks_);}
    int sprintf(const char *fmt, ...) {
        va_list ap;
        va_start(ap, fmt);
        const int ret(kvsprintf(&ks_, fmt, ap));
        va_end(ap);
        return ret;
    }

    // Transfer ownership
    char  *release() {auto ret(ks_.s); ks_.l = ks_.m = 0; ks_.s = nullptr; return ret;}

    // STL imitation
    size_t  size() const {return ks_.l;}
    auto        begin() const {return ks_.s;}
    auto          end() const {return ks_.s + ks_.l;}
    const auto cbegin() const {return const_cast<const char *>(ks_.s);}
    const auto   cend() const {return const_cast<const char *>(ks_.s + ks_.l);}
    void pop() {ks_.s[--ks_.l] = 0;}
    void pop(size_t n) {
        ks_.l = ks_.l > n ? ks_.l - n: 0;
        ks_.s[ks_.l] = 0;
    }

    void clear() {ks_.l = 0;}

    const char     *data() const {return ks_.s;}
    char           *data() {return ks_.s;}
    auto resize(size_t new_size) {
        return ks_resize(&ks_, new_size);
    }

    // std::string imitation: Appending
    // Append char
    auto &operator+=(const char c) {putc(c);  return *this;}

    // Append formatted int/unsigned/long
    auto &operator+=(int c)        {putw(c);  return *this;}
    auto &operator+=(unsigned c)   {putuw(c); return *this;}
    auto &operator+=(long c)       {putl(c);  return *this;}

    // Append string forms
    auto &operator+=(const kstring_t *ks) {
        putsn(ks->s, ks->l);
        return *this;
    }
    auto &operator+=(const kstring_t &ks) {
        return operator+=(&ks);
    }
    auto &operator+=(const std::string &s) {
        putsn(s.data(), s.size());
        return *this;
    }
    auto &operator+=(const KString &other) {return operator+=(other.ks_);}
    auto &operator+=(const char *s)        {puts(s); return *this;}

    // Access
    const char &operator[](size_t index) const {return ks_.s[index];}
    char       &operator[](size_t index)       {return ks_.s[index];}

    int write(FILE *fp) const {return std::fwrite(ks_.s, 1, ks_.l, fp);}
    int write(int fd)   const {return     ::write(fd, ks_.s, ks_.l);}
};

} // namespace ks

#endif // #ifndef _KS_WRAPPER_H__
