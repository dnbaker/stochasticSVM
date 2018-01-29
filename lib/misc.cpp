#include "misc.h"
#include "klib/kstring.h"

namespace svm {

dims_t::dims_t(const char *fn): ns_(0), nd_(-1) {
    int mcols(1<<5), *t((int *)malloc(mcols * sizeof(int)));
    gzFile fp(gzopen(fn, "rb"));
    if(fp == nullptr) throw std::runtime_error(std::string("Could not open file at ") + fn);
    std::string line;
    int c;
    while((c = gzgetc(fp)) != EOF) {
        line += (char)c;
        if(c == '\n') {
            if(line[0] == '#') {
                LOG_DEBUG("Skipping comment line (%s)\n", line.data());
                line.clear();
                continue;
            }
            nd_ = ksplit_core(&line[0], 0, &mcols, &t) - 1;
            ++ns_;
            line.clear();
        }
    }
    LOG_DEBUG("Value of c: %i\n", c);
    LOG_DEBUG("nd: %i. ns: %i\n", c, nd_, ns_);
    free(t);
    gzclose(fp);
    if(nd_ == 0) throw std::runtime_error("nd_ not defined. Abort!");
}

} //namespace svm
