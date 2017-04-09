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
    while(fh.good()) {
        std::getline(fh, line);
        if(line.empty() || line[0] == '#' || line[0] == '\n') continue;
        p = &line[0];
        for(unsigned ind(0); ind < dims.second; ++ind) {
            while(std::isspace(*p)) ++p;
            m(linenum, ind) = atof(p);
            while(!std::isspace(*p)) ++p;
        }
        while(std::isspace(*p)) ++p;
        v[linenum] = atoi(p);
        ++linenum;
    }
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
    for(auto i(0); i < (ssize_t)v.size(); ++i) std::fprintf(stderr, "Value at index %u of Y is %i\n", i, v[i]);
#endif
    return std::pair<DynamicMatrix<MatrixType>, DynamicVector<VectorType>>(std::move(m), std::move(v));
}

} //namespace svm

#endif // _PARSE_H_
