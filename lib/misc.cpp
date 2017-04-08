#include "misc.h"
#include <zlib.h>

namespace svm {

std::pair<size_t, unsigned> count_dims(const char *fn, size_t bufsize) {
    int ncols, mcols(0), *cols(nullptr);
    size_t nlines(0);
    char *buf((char *)calloc(bufsize, sizeof(char))), *line;
    gzFile fp(gzopen(fn, "rb"));
    if(fp == nullptr) throw std::runtime_error(std::string("Could not open file at") + fn);
    line = gzgets(fp, buf, bufsize);
    if(line) ++nlines;
    ncols = ksplit_core(line, 0, &mcols, &cols);
    while((line = gzgets(fp, buf, bufsize))) ++nlines;
    free(buf); free(cols);
    return std::pair<size_t, unsigned>(nlines, ncols - 1); // subtract one for the labels.
}

} //namespace svm
