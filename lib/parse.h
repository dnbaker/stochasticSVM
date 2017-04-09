#ifndef _PARSE_H_
#define _PARSE_H_
#include "lib/misc.h"
#include <iostream>
#include <fstream>
#include <set>
#include <map>

namespace svm {

template<typename MatrixType, typename VectorType>
std::pair<DynamicMatrix<MatrixType>, DynamicVector<VectorType>> parse_problem(const char *fn) {
    std::ios::sync_with_stdio(false);
    std::pair<size_t, unsigned> dims(count_dims(fn));
    std::fprintf(stderr, "rows: %zu. columns: %u\n", dims.first, dims.second);
    DynamicMatrix<MatrixType> m(dims.first, dims.second);
    DynamicVector<VectorType> v(dims.first);
    std::ifstream fh(fn);
    std::string line;
    size_t linenum(0);
    char *p;
    int max(16), *t((int *)malloc(max * sizeof(int)));
    while(fh.good()) {
        std::getline(fh, line);
        if(line.empty() || line[0] == '#' || line[0] == '\n') continue;
        p = &line[0];
        //const auto pc(line.data());
        const int ntoks(ksplit_core(p, 0, &max, &t));
        if(dims.second != ntoks - 1) LOG_EXIT("Expected %i data rows. Found %i\n", dims.second, ntoks - 1);
        for(int i(0); i < ntoks - 1; ++i) m(linenum, i) = atof(p + t[i]);
        v[linenum++] = atoi(p + t[ntoks - 1]);
        if(linenum % 10000 == 0) std::fprintf(stderr, "Parsed %zu lines\n", linenum);
    }
    free(t);
    std::fprintf(stderr, "linenum: %zu. num rows: %zu. cols: %zu.\n", linenum, m.rows(), m.columns());
    assert(linenum == dims.first);
    std::set<VectorType> set(std::begin(v), std::end(v));
    std::vector<VectorType> vec(std::begin(set), std::end(set));
    std::sort(std::begin(vec), std::end(vec));
    std::map<VectorType, int> map;
    int index(0);
    if(vec.size() == 2) {
        map[vec[0]] = -1;
        map[vec[1]] = 1;
    } else for(auto i(std::begin(vec)), e(std::end(vec)); i != e; ++i) map[*i] = ++index;
    for(auto &i: v) i = map[i];
#if !NDEBUG
#endif
    return std::pair<DynamicMatrix<MatrixType>, DynamicVector<VectorType>>(std::move(m), std::move(v));
}

} //namespace svm

#endif // _PARSE_H_
