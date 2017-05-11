#include "lib/linear_svm.h"
#include <getopt.h>
#include <iostream>
using namespace svm;

int usage(char *ex) {
    char buf[1024];
    std::sprintf(buf, "Usage: %s <opts> data\n"
                       "Flags:\n-p:\tNumber of processes [1]\n\n"
                       "-[h?]:\tHelp menu.\n"
                       "-l:\tSet lambda parameter.\n"
                       "-E:\tSet eta parameter (NORMA and Zhang algorithms only).\n"
                       "-e:\tSet epsilon parameter (for termination). Set to 0 to always execute <max_iter> times.\n"
                       "-M:\tMax iter (100000)\n"
                       "-b:\tBatch size\n"
                       "-s:\tNumber of dimensions for sparse parsing. Also determines the use of sparse rather than dense parsing.\n"
                       "-N:\tUse Norma learning rate. Requires -e.\n"
                       "-F:\tUse fixed learning rate. Requires -e.\n"
                 , ex);
    cerr << buf;
    return EXIT_FAILURE;
}

enum Policy:size_t{
    PEGASOS = 0,
    NORMA   = 1,
    FIXED   = 2
};

#define RUN_SVM \
        svm.train();\
        svm.write(ofp);\
        if(argc > optind + 1) {\
            size_t nlines(0), nerror(0);\
            DynamicVector<FLOAT_TYPE> vec(svm.ndims());\
            vec[vec.size() - 1] = 1.;\
            std::ifstream is(argv[optind + 1]);\
            int label;\
            for(std::string line;std::getline(is, line);) {\
                /*cerr << line << '\n';*/\
                vec = 0.; vec[vec.size() - 1] = 1.;\
                label = atoi(line.data());\
                char *p(line.data());\
                while(!std::isspace(*p)) ++p;\
                for(;;) {\
                    while(*p == '\t' || *p == ' ') ++p;\
                    const int ind(atoi(p) - 1);\
                    p = strchr(p, ':');\
                    if(p) ++p;\
                    else throw std::runtime_error("No ':' found!");\
                    vec[ind] = atof(p);\
                    while(!std::isspace(*p)) ++p;\
                    if(*p == '\n' || *p == '\0' || p > line.data() + line.size()) break;\
                }\
                /*cerr << vec;*/\
                nerror += (svm.classify(vec) != label);\
                ++nlines;\
            }\
            cerr << "test error rate: " << 100. * nerror / nlines << "%\n";\
        }

int main(int argc, char *argv[]) {
    int c, batch_size(256), nd_sparse(0);
    FLOAT_TYPE lambda(0.5), eta(0.0), eps(1e-6);
    size_t max_iter(100000);
    unsigned nthreads(1);
    std::ios::sync_with_stdio(false);
    FILE *ofp(stdout);
    Policy policy(PEGASOS);
    for(char **p(argv + 1); *p; ++p) if(strcmp(*p, "--help") == 0) goto usage;
    while((c = getopt(argc, argv, "E:e:M:s:p:b:l:o:FNh?")) >= 0) {
        switch(c) {
            case 'E': eta        = atof(optarg); break;
            case 'e': eps        = atof(optarg); break;
            case 'p': nthreads   = atoi(optarg); break;
            case 'M': max_iter   = strtoull(optarg, 0, 10); break;
            case 'b': batch_size = atoi(optarg); break;
            case 'l': lambda     = atof(optarg); break;
            case 's': nd_sparse  = atoi(optarg); break;
            case 'N': policy     = NORMA;        break; // Guerra, Guerra!
            case 'F': policy     = FIXED;        break;
            case 'o': ofp        = fopen(optarg, "w");
                if(ofp == nullptr) throw std::runtime_error(
                    std::string("Could not open file at ") + optarg);
                break;
            case 'h': case '?': usage: return usage(*argv);
        }
    }
    if((policy == NORMA || policy == FIXED) && eta == 0.0) {
        throw std::runtime_error(
            "eta (-E) must be set for Norma or Fixed Learning rate policies.");
    }

    if(optind == argc) goto usage;
    blaze::setNumThreads(nthreads);
    PegasosLearningRate<FLOAT_TYPE> lp(lambda);
    NormaLearningRate<FLOAT_TYPE>  nlp(eta);
    FixedLearningRate<FLOAT_TYPE>  flp(eta);
    if(policy == NORMA) {
        LinearSVM<FLOAT_TYPE, DynamicMatrix<FLOAT_TYPE>, decltype(nlp)> svm =
            nd_sparse ? LinearSVM<FLOAT_TYPE, DynamicMatrix<FLOAT_TYPE>, decltype(nlp)>(argv[optind], nd_sparse, lambda, nlp, batch_size, max_iter, eps)
                      : LinearSVM<FLOAT_TYPE, DynamicMatrix<FLOAT_TYPE>, decltype(nlp)>(argv[optind], lambda, nlp, batch_size, max_iter, eps);
        RUN_SVM
    } else if(policy == FIXED) {
        LinearSVM<FLOAT_TYPE, DynamicMatrix<FLOAT_TYPE>, decltype(flp)> svm =
            nd_sparse ? LinearSVM<FLOAT_TYPE, DynamicMatrix<FLOAT_TYPE>, decltype(flp)>(argv[optind], nd_sparse, lambda, flp, batch_size, max_iter, eps)
                      : LinearSVM<FLOAT_TYPE, DynamicMatrix<FLOAT_TYPE>, decltype(flp)>(argv[optind], lambda, flp, batch_size, max_iter, eps);
        RUN_SVM
    } else {
        LinearSVM<FLOAT_TYPE, DynamicMatrix<FLOAT_TYPE>, decltype(lp)> svm =
            nd_sparse ? LinearSVM<FLOAT_TYPE, DynamicMatrix<FLOAT_TYPE>, decltype(lp)>(argv[optind], nd_sparse, lambda, lp, batch_size, max_iter, eps)
                      : LinearSVM<FLOAT_TYPE, DynamicMatrix<FLOAT_TYPE>, decltype(lp)>(argv[optind], lambda, lp, batch_size, max_iter, eps);
        RUN_SVM
    }
    if(ofp != stdout) fclose(ofp);
}
