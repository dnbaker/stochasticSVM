#ifndef _PARSE_H_
#define _PARSE_H_
#include "lib/misc.h"
#include <iostream>
#include <fstream>
#include <set>
#include <map>

namespace svm {

template<typename MatrixType, typename VectorType>
std::pair<DynamicMatrix<MatrixType>, DynamicVector<VectorType>> parse_problem(const char *fn, const size_t ns, const size_t nd) {
    gzFile fp(gzopen(fn, "rb"));
    if(fp == nullptr) throw std::runtime_error(std::string("Could not open file at ") + fn);
    std::vector<char> line;
    line.reserve(1 << 12);
    size_t linenum(0);
    char *p;
#if !NDEBUG
    char *line_end;
#endif
    int c;
    DynamicMatrix<MatrixType> m(ns, nd);
    DynamicVector<VectorType> v(ns);
    while((c = gzgetc(fp)) != EOF) {
        if(c != '\n') {
            line.push_back(c);
            continue;
        }
        line.push_back('\0');
        if(line[0] == '#' || line[0] == '\n') {
            line.resize(0);
            continue;
        }
        p = &line[0];
#if !NDEBUG
        line_end = &line[line.size() - 1];
#endif
        unsigned ind(0);
        for(ind = 0; ind < nd; ++ind) {
            while(std::isspace(*p)) ++p;
            m(linenum, ind) = atof(p);
            while(!std::isspace(*p)) ++p;
            assert(p < line_end);
        }
        while(std::isspace(*p)) ++p;
        v[linenum] = atoi(p);
        ++linenum;
        if(!(linenum & 255)) LOG_DEBUG("%zu lines processed\n", linenum);
    }
    for(auto i(0); i < m.rows(); ++i) for(auto j(0); j < m.columns(); ++j) LOG_DEBUG("ZOMGZ %i, %i has %f\n", i, j, m(i, j));
    gzclose(fp);
    LOG_DEBUG("linenum: %zu. num rows: %zu. cols: %zu.\n", linenum, m.rows(), m.columns());
    assert(linenum == ns);
    return std::pair<DynamicMatrix<MatrixType>, DynamicVector<VectorType>>(std::move(m), std::move(v));
}

template<typename MatrixType, typename VectorType>
std::pair<DynamicMatrix<MatrixType>, DynamicVector<VectorType>> parse_problem(const char *fn) {
    size_t nd, ns;
    std::tie(ns, nd) = count_dims(fn);
    return parse_problem<MatrixType, VectorType>(fn, ns, nd);
}

} //namespace svm

#endif // _PARSE_H_
