#ifndef _PARSE_H_
#define _PARSE_H_
#include "lib/misc.h"
#include <iostream>
#include <fstream>
#include <set>
#include <map>

namespace svm {

template<typename MatrixType, typename VectorType>
std::tuple<DynamicMatrix<MatrixType>, DynamicVector<VectorType>, std::unordered_map<VectorType, std::string>> parse_problem(const char *fn, const dims_t &dims) {
    std::unordered_map<std::string, VectorType> name_map;
    std::vector<std::string> names;
    gzFile fp(gzopen(fn, "rb"));
    if(fp == nullptr) throw std::runtime_error(std::string("Could not open file at ") + fn);
    std::vector<char> line;
    line.reserve(1 << 12);
    size_t linenum(0);
    char *p;
    LOG_DEBUG("ns: %zu. nd: %zu\n", dims.ns_, dims.nd_);
#if !NDEBUG
    char *line_end;
#endif
    int c;
    DynamicMatrix<MatrixType> m(dims.ns_, dims.nd_);
    DynamicVector<VectorType> v(dims.ns_);
    std::string class_name;
    VectorType  class_id(0);
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
        for(ind = 0; ind < dims.nd_; ++ind) {
            while(std::isspace(*p)) ++p;
            m(linenum, ind) = atof(p);
            while(!std::isspace(*p)) ++p;
            assert(p < line_end);
        }
        while(std::isspace(*p)) ++p;
        class_name = p;
        auto m(name_map.find(class_name));
        if(m == name_map.end()) {
            name_map.emplace(class_name, class_id++);
            v[linenum] = class_id - 1;
        } else v[linenum] = m->second;
        ++linenum;
        //if(!(linenum & 255)) LOG_DEBUG("%zu lines processed\n", linenum);
        line.clear();
    }
    //cerr << "Matrix is \n" << m;
    gzclose(fp);
    LOG_DEBUG("linenum: %zu. num rows: %zu. cols: %zu.\n", linenum, m.rows(), m.columns());
    assert(linenum == dims.ns_);
    std::unordered_map<VectorType, std::string> class_name_map;
    for(auto &pair: name_map) class_name_map.emplace(pair.second, pair.first);
    return std::tuple<DynamicMatrix<MatrixType>, DynamicVector<VectorType>, std::unordered_map<VectorType, std::string>>(std::move(m), std::move(v), std::move(class_name_map));
}

template<typename MatrixType, typename VectorType>
std::pair<DynamicMatrix<MatrixType>, DynamicVector<VectorType>> parse_problem(const char *fn) {
    return parse_problem<MatrixType, VectorType>(fn, dims_t(fn));
}

} //namespace svm

#endif // _PARSE_H_
