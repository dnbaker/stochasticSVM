#ifndef _PARSE_H_
#define _PARSE_H_
#include "lib/misc.h"
#include <iostream>
#include <fstream>
#include <set>
#include <map>

namespace svm {

template<typename MatrixType, typename VectorType>
auto parse_problem(const char *fn, const dims_t &dims) {
    std::unordered_map<std::string, VectorType> name_map;
    std::vector<std::string> names;
    gzFile fp(gzopen(fn, "rb"));
    if(fp == nullptr) throw std::runtime_error(std::string("Could not open file at ") + fn);
    ks::KString line;
    line.resize(1 << 12);
    size_t linenum(0);
    char *p;
    LOG_DEBUG("ns: %zu. nd: %zu\n", dims.ns_, dims.nd_);
#if !NDEBUG
    char *line_end;
#endif
    int c;
    DynamicMatrix<MatrixType> m(dims.ns_, dims.nd_ + 1);
    DynamicVector<VectorType> v(dims.ns_);
    std::string class_name;
    VectorType  class_id(0);
    while((c = gzgetc(fp)) != EOF) {
        if(c != '\n') {
            line.putc_(c);
            continue;
        }
        line.terminate();
        if(line[0] == '#' || line[0] == '\n') {
            line.clear();
            continue;
        }
        p = line.data();
#if !NDEBUG
        line_end = &line[line.size() - 1];
#endif
        unsigned ind(0);
        for(ind = 0; ind < dims.nd_; ++ind) {
            while(std::isspace(*p)) ++p;
            m(linenum, ind) = std::atof(p);
            while(!std::isspace(*p)) ++p;
#if !NDEBUG
            if(p >= line_end) {
                fprintf(stderr, "line: %s. Line number: %zu. Line length: %zu. token number: %u\n", line.data(), linenum, line_end - line.data(), ind);
            }
            assert(p < line_end);
#endif
        }
        while(std::isspace(*p)) ++p;
        class_name = p;
        auto m(name_map.find(class_name));
        if(m == name_map.end()) {
            name_map.emplace(class_name, class_id++);
            v[linenum] = class_id - 1;
        } else v[linenum] = m->second;
        ++linenum;
        line.clear();
    }
    gzclose(fp);
    LOG_DEBUG("linenum: %zu. num rows: %zu. cols: %zu.\n", linenum, m.rows(), m.columns());
    assert(linenum == dims.ns_);
    using CNMapType = std::unordered_map<VectorType, std::string>;
    CNMapType class_name_map;
    for(auto &pair: name_map) class_name_map.emplace(pair.second, pair.first);
    return std::make_tuple(std::move(m), std::move(v), std::move(class_name_map));
}

template<typename MatrixType, typename VectorType>
std::pair<DynamicMatrix<MatrixType>, DynamicVector<VectorType>> parse_problem(const char *fn) {
    return parse_problem<MatrixType, VectorType>(fn, dims_t(fn));
}

} //namespace svm

#endif // _PARSE_H_
