#include "lib/problem.h"
#include <getopt.h>
using namespace svm;

int main(int argc, char *argv[]) {
    int c;
    double kappa(0.01);
    double kc(-0.2);
    unsigned nthreads(std::thread::hardware_concurrency());
    while((c = getopt(argc, argv, "w:M:S:p:k:f:h?")) >= 0) {
        switch(c) {
            case 'p': nthreads = atoi(optarg); break;
            case 'k': kappa    = atof(optarg); break;
            case 'c': kc       = atof(optarg); break;
        }
    }
    blaze::setNumThreads(nthreads);
    LOG_ASSERT(blaze::getNumThreads() == nthreads);
    SVM<LinearKernel<double>> svm(argv[optind], 0.4, 256);
#if 0
    auto row1(row(pair.first, 1));
    auto row2(row(pair.first, 2));
    RBFKernel<double>           gk(0.2);
    TanhKernel<double>          tk(0.2, 0.4);
#endif
//#if GENERATE_TANH_KERNEL
#if 1
    TanhKernelMatrix<double>   tkm(kappa, kc);
    DynamicMatrix<double> kernel_matrix(tkm(svm.get_matrix()));
    for(size_t i(0), e(kernel_matrix.rows()); i != e; ++i) { for(size_t j(0), f(kernel_matrix.columns()); j != f; ++j) {
        std::fprintf(stderr, "km value at i is %f\n", kernel_matrix(i, j));
    } }
    LinearKernel<double> lk;
    float zomg(0.);
    auto row1(row(svm.get_matrix(), 0));
    auto row2(row(svm.get_matrix(), 1));
    std::fprintf(stderr, "Kernel result: %f\n", lk(row1, row2));
    for(u32 i(0); i < row1.size(); ++i) zomg += row1[i] * row2[i];
    std::fprintf(stderr, "Kernel result: %f\n", zomg);
#endif
}
