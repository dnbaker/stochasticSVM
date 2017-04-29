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
                 , ex);
    cerr << buf;
    return EXIT_FAILURE;
}

int main(int argc, char *argv[]) {
    int c;
    double kappa(0.0025);
    double kc(-0.25);
    unsigned nthreads(1);
    char cerr_buf[1 << 16];
    cerr.rdbuf()->pubsetbuf(cerr_buf, sizeof cerr_buf);
    std::ios::sync_with_stdio(false);
    for(char **p(argv + 1); *p; ++p) if(strcmp(*p, "--help") == 0) goto usage;
    while((c = getopt(argc, argv, "c:w:M:S:p:k:f:h?")) >= 0) {
        switch(c) {
            case 'p': nthreads = atoi(optarg); break;
            case 'k': kappa    = atof(optarg); break;
            case 'c': kc       = atof(optarg); break;
            case 'h': case '?': usage: return usage(*argv);
        }
    }

    if(optind == argc) goto usage;

    blaze::setNumThreads(nthreads);
    LOG_ASSERT(blaze::getNumThreads() == nthreads);
    LinearKernel<double> linear_kernel;
    SVMTrainer<LinearKernel<double>, double> svm(argv[optind], 0.4, linear_kernel, 256);
    svm.train_linear();
    // cerr << "Matrix in: \n" << svm.get_matrix();
    cerr << "Frobenius norm of matrix is " << frobenius_norm(svm.get_matrix()) << '\n';
    RBesselKernel<double> rbk(0.1);
#if 0
    auto row1(row(pair.first, 1));
    auto row2(row(pair.first, 2));
    RBFKernel<double>           gk(0.2);
    TanhKernel<double>          tk(0.2, 0.4);
#endif
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
}
