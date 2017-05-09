#include "lib/svm.h"
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
#if CERR_BUFF
    char cerr_buf[1 << 16];
    cerr.rdbuf()->pubsetbuf(cerr_buf, sizeof cerr_buf);
    cerr << std::nounitbuf;
#endif
    std::ios::sync_with_stdio(false);
    for(char **p(argv + 1); *p; ++p) if(strcmp(*p, "--help") == 0) goto usage;
    while((c = getopt(argc, argv, "c:M:s:p:k:b:l:h?")) >= 0) {
        switch(c) {
            case 'p': nthreads   = atoi(optarg); break;
            case 'k': kappa      = atof(optarg); break;
            case 'c': kc         = atof(optarg); break;
            case 'M': max_iter   = strtoull(optarg, 0, 10); break;
            case 'b': batch_size = atoi(optarg); break;
            case 'l': lambda     = atof(optarg); break;
            case 's': nd_sparse  = atoi(optarg); 
                cerr << "Sparse value set: " << nd_sparse << '\n';
                break;
            case 'h': case '?': usage: return usage(*argv);
        }
    }

    if(optind == argc) goto usage;

    blaze::setNumThreads(nthreads);
    LOG_ASSERT(blaze::getNumThreads() == nthreads);
    LinearKernel<double> linear_kernel;
    PegasosLearningRate<double> lp(lambda);
    if(nd_sparse) cerr << "nd sparse " << nd_sparse << '\n';
    SVMTrainer<LinearKernel<double>, double, DynamicMatrix<double>, int, decltype(lp)> svm =
        nd_sparse ? SVMTrainer<LinearKernel<double>, double, DynamicMatrix<double>, int, decltype(lp)>(argv[optind], nd_sparse, lambda, lp, linear_kernel, batch_size, max_iter)
                  : SVMTrainer<LinearKernel<double>, double, DynamicMatrix<double>, int, decltype(lp)>(argv[optind], lambda, lp, linear_kernel, batch_size, max_iter);
    svm.train_linear();
    // cerr << "Matrix in: \n" << svm.get_matrix();
    //cerr << "Frobenius norm of matrix is " << frobenius_norm(svm.get_matrix()) << '\n';
    RBesselKernel<double> rbk(0.1);
#if 0
    auto row1(row(pair.first, 1));
    auto row2(row(pair.first, 2));
    RBFKernel<double>           gk(0.2);
    TanhKernel<double>          tk(0.2, 0.4);
    TanhKernelMatrix<double>   tkm(kappa, kc);
    DynamicMatrix<double> kernel_matrix(tkm(svm.get_matrix()));
    // cout << kernel_matrix;
    LinearKernel<double> lk;
    double zomg(0.);
    auto row1(row(svm.get_matrix(), 0));
    auto row2(row(svm.get_matrix(), 1));
    auto ret(rbk(row1, row2));
    cerr << "RBK result: " << ret << '\n';
    cerr << "Kernel result: " << lk(row1, row2) << '\n';
    for(u32 i(0); i < row1.size(); ++i) zomg += row1[i] * row2[i];
    cerr << "Manual result: " << zomg << '\n';
#endif
}
