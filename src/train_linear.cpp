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
                 , ex);
    cerr << buf;
    return EXIT_FAILURE;
}

int main(int argc, char *argv[]) {
    int c, batch_size(1), nd_sparse(0);
    double kappa(0.0025);
    double kc(-0.25);
    double lambda(0.5);
    size_t max_iter(100000);
    unsigned nthreads(1);
    std::ios::sync_with_stdio(false);
    FILE *ofp(stdout);
    for(char **p(argv + 1); *p; ++p) if(strcmp(*p, "--help") == 0) goto usage;
    while((c = getopt(argc, argv, "c:M:s:p:k:b:l:o:h?")) >= 0) {
        switch(c) {
            case 'p': nthreads   = atoi(optarg); break;
            case 'k': kappa      = atof(optarg); break;
            case 'c': kc         = atof(optarg); break;
            case 'M': max_iter   = strtoull(optarg, 0, 10); break;
            case 'b': batch_size = atoi(optarg); break;
            case 'l': lambda     = atof(optarg); break;
            case 's': nd_sparse  = atoi(optarg); break;
            case 'h': case '?': usage: return usage(*argv);
            case 'o': ofp = fopen(optarg, "w"); break;
        }
    }

    if(optind == argc) goto usage;

    blaze::setNumThreads(nthreads);
    assert(blaze::getNumThreads() == nthreads);
    LinearKernel<double> linear_kernel;
    PegasosLearningRate<double> lp(lambda);
    LinearSVM<LinearKernel<double>, double, DynamicMatrix<double>, int, decltype(lp)> svm =
        nd_sparse ? LinearSVM<LinearKernel<double>, double, DynamicMatrix<double>, int, decltype(lp)>(argv[optind], nd_sparse, lambda, lp, linear_kernel, batch_size, max_iter)
                  : LinearSVM<LinearKernel<double>, double, DynamicMatrix<double>, int, decltype(lp)>(argv[optind], lambda, lp, linear_kernel, batch_size, max_iter);
    svm.train();
    svm.write(ofp);
    if(ofp != stdout) fclose(ofp);
}
