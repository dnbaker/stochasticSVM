#include "lib/kernel_svm.h"
#include <getopt.h>
#include <iostream>
using namespace svm;

#ifndef NOTIFICATION_INTERVAL
#define NOTIFICATION_INTERVAL 256uL
#endif

static int get_max_ind(const char *fn) {
    gzFile fp(gzopen(fn, "rb"));
    if(fp == nullptr) throw std::runtime_error(("Could not open file at "s + fn).data());
    char buf[1 << 16];
    char *p;
    int tmp, cmax(0);
    while((p = gzgets(fp, buf, sizeof buf))) {
        char *q(strchr(p, '\0'));
        if(q == nullptr) throw std::runtime_error(("Ill-formatted file at "s + fn).data());
        while(*q != ':' && q != p) --q;
        if(q == p) continue;
        while(!std::isspace(*q)) --q;
        if((tmp = atoi(++q)) > cmax) cmax = tmp;
    }
    gzclose(fp);
    return cmax;
}

static int get_max_ind(const char *fn1, const char *fn2) {
    return fn2 ? std::max(get_max_ind(fn1), get_max_ind(fn2)): get_max_ind(fn1);
}

// Macro for running SVM and testing it.

#define RUN_SVM_MATRIX(MatrixKind) \
        svm.train();\
        svm.write(ofp);\
        if(argc > optind + 1) {\
            int moffsets(svm.get_ndims() + 1), *offsets(static_cast<int *>(malloc(moffsets * sizeof(int))));\
            IntCounter counter;\
            size_t nlines(0), nerror(0);\
            MatrixKind<FLOAT_TYPE> vecmat(1, svm.get_ndims());\
            auto vec(row(vecmat, 0));\
            if(svm.get_bias()) vec[vec.size() - 1] = 1.;\
            std::ifstream is(argv[optind + 1]);\
            int label;\
            for(std::string line;std::getline(is, line);) {\
                /*std::cerr << line << '\n';*/\
                vec.reset();\
                if(svm.get_bias()) vec[vec.size() - 1] = 1.;\
                const int ntoks(ksplit_core(static_cast<char *>(&line[0]), 0, &moffsets, &offsets));\
                label = atoi(line.data());\
                for(int i(1); i < ntoks; ++i) {\
                    const char *p(line.data() + offsets[i]);\
                    vec[atoi(p) - 1] = std::atof(strchr(p, ':') + 1);\
                }\
                /*std::cerr << vec;*/\
                if(svm.classify_external(vec) != label) {\
                    ++nerror, counter.add(label);\
                }\
                if(++nlines % NOTIFICATION_INTERVAL == 0) std::cerr << "Processed " << nlines << " lines.\n";\
            }\
            std::free(offsets);\
            cout << "Test error rate: " << 100. * nerror / nlines << "%\n";\
            cout << "Mislabeling: " << counter.str() << '\n';\
        }

#define RUN_DENSE_SVM  RUN_SVM_MATRIX(DynamicMatrix)
#define RUN_SVM        RUN_DENSE_SVM
#define RUN_SPARSE_SVM RUN_SVM_MATRIX(CompressedMatrix)

/* 
 * Giant macro creating an executable for a given kernel.
 * This should help maintainability for multiple kernels.
 */


#define DECLARE_KERNEL_SVM(KERNEL_INIT, KERNEL_ARGS, KERNEL_PARAMS, KERNEL_USAGE, KERNEL_GETOPT)\
\
int usage(char *ex) {\
    char buf[1024];\
    std::sprintf(buf, "Usage: %s <opts> data\n"\
                       "Flags:\n\n-p:\tNumber of processes [1]\n"\
                       "-l:\tSet lambda parameter. [0.5]\n"\
                       KERNEL_USAGE \
                       "-M:\tMax iter [100000]\n"\
                       "-b:\tBatch size [256]\n"\
                       "-e:\tSet epsilon for termination. [1e-6]\n"\
                       "-p:\tSet number of threads. [1]\n"\
                       "-s:\tNumber of dimensions for sparse parsing. Also determines the use of sparse rather than dense parsing. [Set to -1 to infer from file(s)]\n"\
                       "-[h?]:\tHelp menu.\n"\
                 , ex);\
    std::cerr << buf;\
    return EXIT_FAILURE;\
}\
\
int main(int argc, char *argv[]) {\
    int c, batch_size(256), nd_sparse(0);\
    FLOAT_TYPE lambda(0.5), eps(1e-6);\
    KERNEL_PARAMS\
    size_t max_iter(100000);\
    unsigned nthreads(1);\
    std::FILE *ofp(stdout);\
    bool rescale(false), use_sparse(false), bias(true);\
    while((c = getopt(argc, argv, KERNEL_GETOPT "5:e:M:s:p:b:l:o:Brh?")) >= 0) {\
        switch(c) {\
            case '5': use_sparse = true;         break;\
            case 'B': bias       = false;        break;\
            case 'e': eps        = std::atof(optarg); break;\
            case 'p': nthreads   = std::atoi(optarg); break;\
            case 'M': max_iter   = std::strtoull(optarg, 0, 10); break;\
            case 'b': batch_size = std::atoi(optarg); break;\
            case 'l': lambda     = std::atof(optarg); break;\
            case 's': nd_sparse  = std::atoi(optarg); break;\
            case 'r': rescale    = true; break;\
            KERNEL_ARGS\
            case 'o': ofp        = fopen(optarg, "w"); break;\
                if(ofp == nullptr) throw std::runtime_error(\
                    std::string("Could not open file at ") + optarg);\
                break;\
            case 'h': case '?': usage: return usage(*argv);\
        }\
    }\
    KERNEL_INIT;\
    static_assert(sizeof(kernel) == sizeof(kernel));\
    if(nd_sparse < 0) get_max_ind(argv[optind], argv[optind + 1]);\
\
    if(optind == argc) goto usage;\
    blaze::setNumThreads(nthreads);\
    omp_set_num_threads(nthreads);\
    std::cerr << "Training data at " << argv[optind] << ".\n";\
    if(optind < argc - 1) std::cerr << "Test data at " << argv[optind + 1] << ".\n";\
    if(use_sparse) {\
        using OtherType = ::svm::KernelSVM<decltype(kernel), FLOAT_TYPE, ::blaze::CompressedMatrix<FLOAT_TYPE>>;\
        OtherType svm(\
                nd_sparse ? OtherType(argv[optind], nd_sparse, lambda, kernel, batch_size, max_iter, eps, rescale, bias)\
                          : OtherType(argv[optind], lambda, kernel, batch_size, max_iter, eps, rescale, bias));\
        RUN_SPARSE_SVM;\
    } else { \
        using OtherType = ::svm::KernelSVM<decltype(kernel), FLOAT_TYPE>;\
        OtherType svm(\
                nd_sparse ? OtherType(argv[optind], nd_sparse, lambda, kernel, batch_size, max_iter, eps, rescale, bias)\
                          : OtherType(argv[optind], lambda, kernel, batch_size, max_iter, eps, rescale, bias));\
        RUN_DENSE_SVM;\
    }\
    if(ofp != stdout) fclose(ofp);\
}

//#define DECLARE_KERNEL_SVM(KERNEL_INIT, KERNEL_ARGS, KERNEL_PARAMS, KERNEL_USAGE, KERNEL_GETOPT) \
//    DECLARE_KERNEL_SVM_NAME(main, KERNEL_INIT, KERNEL_ARGS, KERNEL_PARAMS, KERNEL_USAGE, KERNEL_GETOPT)
