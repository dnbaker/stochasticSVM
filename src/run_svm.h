#include <getopt.h>
#include <iostream>
using namespace svm;

class IntCounter {
    std::map<int, int> map_;
public:
    void add(int val) {
        ++map_[val];
    }
    std::string str() const {
        std::string ret("{");
        for(auto &pair: map_) ret += std::to_string(pair.first) + ": " + std::to_string(pair.second) + ", ";
        ret.pop_back();
        ret[ret.size() - 1] = '}';
        return ret;
    }
};

#define RUN_SVM \
        svm.train();\
        svm.write(ofp);\
        if(argc > optind + 1) {\
            int moffsets(svm.get_ndims() + 1), *offsets(static_cast<int *>(malloc(moffsets * sizeof(int))));\
            IntCounter counter;\
            size_t nlines(0), nerror(0);\
            DynamicMatrix<FLOAT_TYPE> vecmat(1, svm.get_ndims());\
            auto vec(row(vecmat, 0));\
            if(svm.get_bias()) vec[vec.size() - 1] = 1.;\
            std::ifstream is(argv[optind + 1]);\
            int label;\
            for(std::string line;std::getline(is, line);) {\
                /*cerr << line << '\n';*/\
                vec = 0.;\
                if(svm.get_bias()) vec[vec.size() - 1] = 1.;\
                const int ntoks(ksplit_core(static_cast<char *>(&line[0]), 0, &moffsets, &offsets));\
                label = atoi(line.data());\
                for(int i(1); i < ntoks; ++i) {\
                    const char *p(line.data() + offsets[i]);\
                    vec[atoi(p) - 1] = atof(strchr(p, ':') + 1);\
                }\
                /*cerr << vec;*/\
                if(svm.classify_external(vec) != label) {\
                    ++nerror, counter.add(label);\
                }\
                ++nlines;\
            }\
            std::free(offsets);\
            cout << "Test error rate: " << 100. * nerror / nlines << "%\n";\
            cout << "Mislabeling: " << counter.str() << '\n';\
        }

