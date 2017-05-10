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
                       "-e:\tSet eta parameter (NORMA and Zhang algorithms only).\n"
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

int main(int argc, char *argv[]) {
    int c, batch_size(256), nd_sparse(0);
    FLOAT_TYPE lambda(0.5), eta(0.0);
    size_t max_iter(100000);
    unsigned nthreads(1);
    std::ios::sync_with_stdio(false);
    FILE *ofp(stdout);
    Policy policy(PEGASOS);
    for(char **p(argv + 1); *p; ++p) if(strcmp(*p, "--help") == 0) goto usage;
    while((c = getopt(argc, argv, "e:M:s:p:b:l:o:FNh?")) >= 0) {
        switch(c) {
            case 'e': eta        = atof(optarg); break;
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
            "eta (-e) must be set for Norma or Fixed Learning rate policies.");
    }

    if(optind == argc) goto usage;
    blaze::setNumThreads(nthreads);
    PegasosLearningRate<FLOAT_TYPE> lp(lambda);
    NormaLearningRate<FLOAT_TYPE>  nlp(eta);
    FixedLearningRate<FLOAT_TYPE>  flp(eta);
    if(policy == NORMA) {
        LinearSVM<FLOAT_TYPE, DynamicMatrix<FLOAT_TYPE>, int, decltype(nlp)> svm =
            nd_sparse ? LinearSVM<FLOAT_TYPE, DynamicMatrix<FLOAT_TYPE>, int, decltype(nlp)>(argv[optind], nd_sparse, lambda, nlp, batch_size, max_iter)
                      : LinearSVM<FLOAT_TYPE, DynamicMatrix<FLOAT_TYPE>, int, decltype(nlp)>(argv[optind], lambda, nlp, batch_size, max_iter);
        svm.train();
        svm.write(ofp);
    } else if(policy == FIXED) {
        LinearSVM<FLOAT_TYPE, DynamicMatrix<FLOAT_TYPE>, int, decltype(flp)> svm =
            nd_sparse ? LinearSVM<FLOAT_TYPE, DynamicMatrix<FLOAT_TYPE>, int, decltype(flp)>(argv[optind], nd_sparse, lambda, flp, batch_size, max_iter)
                      : LinearSVM<FLOAT_TYPE, DynamicMatrix<FLOAT_TYPE>, int, decltype(flp)>(argv[optind], lambda, flp, batch_size, max_iter);
        svm.train();
        svm.write(ofp);
    } else {
        LinearSVM<FLOAT_TYPE, DynamicMatrix<FLOAT_TYPE>, int, decltype(lp)> svm =
            nd_sparse ? LinearSVM<FLOAT_TYPE, DynamicMatrix<FLOAT_TYPE>, int, decltype(lp)>(argv[optind], nd_sparse, lambda, lp, batch_size, max_iter)
                      : LinearSVM<FLOAT_TYPE, DynamicMatrix<FLOAT_TYPE>, int, decltype(lp)>(argv[optind], lambda, lp, batch_size, max_iter);
        svm.train();
        svm.write(ofp);
    }
    if(ofp != stdout) fclose(ofp);
}
