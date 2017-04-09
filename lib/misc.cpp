#include "misc.h"
#include <zlib.h>
#include <string>

namespace svm {

std::pair<size_t, unsigned> count_dims(const char *fn, size_t bufsize) {
    size_t nlines(0);
    int mcols(1<<10), *t((int *)malloc(mcols * sizeof(int)));
    gzFile fp(gzopen(fn, "rb"));
    std::string line;
    size_t ncols(0);
    int c;
    while((c = gzgetc(fp)) != EOF) {
        line += (char)c;
        if(c == '\n') {
            ncols = ksplit_core(&line[0], 0, &mcols, &t);
            ++nlines;
            break;
        }
    }
    while((c = gzgetc(fp)) != EOF) nlines += (c == '\n');
    return std::pair<size_t, unsigned>(nlines, ncols - 1); // subtract one for the labels.
}

} //namespace svm
