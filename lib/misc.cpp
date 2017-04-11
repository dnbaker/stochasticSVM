#include "misc.h"
#include <zlib.h>
#include <string>

namespace svm {

dims_t::dims_t(const char *fn): ns_(0), nd_(-1) {
    int mcols(1<<5), *t((int *)malloc(mcols * sizeof(int)));
    gzFile fp(gzopen(fn, "rb"));
    std::string line;
    int c;
    while((c = gzgetc(fp)) != EOF) {
        line += (char)c;
        if(c == '\n') {
            nd_ = ksplit_core(&line[0], 0, &mcols, &t) - 1;
            ++ns_;
            line.clear();
            break;
        }
    }
    if(nd_ + 1 == 0) throw std::runtime_error("nd_ not defined. Abort!");
    free(t);
    while((c = gzgetc(fp)) != EOF) ns_ += (c == '\n');
    gzclose(fp);
}

} //namespace svm
