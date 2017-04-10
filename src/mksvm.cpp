#include "lib/problem.h"
#include <getopt.h>
using namespace svm;

int main(int argc, char *argv[]) {
    int c, nthreads(std::thread::hardware_concurrency());
    while((c = getopt(argc, argv, "w:M:S:p:k:T:F:tfHh?")) >= 0) {
        switch(c) {
            case 'p': nthreads = atoi(optarg); break;
        }
    }
    blaze::setNumThreads(nthreads);
    LOG_ASSERT(blaze::getNumThreads() == nthreads);
    SVM<LinearKernel<double>> svm(argv[optind], 0.4, 256);
#if 0
    auto row1(row(pair.first, 1));
    auto row2(row(pair.first, 2));
    LinearKernel<float> lk;
    RBFKernel<float>           gk(0.2);
    TanhKernel<float>          tk(0.2, 0.4);
#endif
//#if GENERATE_TANH_KERNEL
#if 1
    TanhKernelMatrix<float>   tkm(0.2, 0.4);
    //DynamicMatrix<float> kernel_matrix(tkm(pair.first));
#endif
#if 0
    std::fprintf(stderr, "Kernel result: %f\n", lk(row1, row2));
    float zomg(0);
    for(u32 i(0); i < row1.size(); ++i) zomg += row1[i] * row2[i];
    std::fprintf(stderr, "Kernel result: %f\n", zomg);
    // Just testing blaze.
    DynamicMatrix<float> m(4000, 10);
    DynamicMatrix<float> n(10, 4000);
    std::memset(&m(0, 0), 0, sizeof(float) * m.capacity());
    std::fprintf(stderr, "Rows: %zu. Columns: %zu. Capacity: %zu. Nonzeros: %zu\n",
                 m.rows(), m.columns(), m.capacity(), m.nonZeros());
    for(auto i(0l); i < (long)m.rows(); ++i) {
        for(auto j(0l); j < (long)m.columns(); ++j) {
            m(i, j) = i * j * j - j * i * i;
            n(j, i) = -i + j * (i + j);
            std::fprintf(stderr, "Element at row %lu and column %lu is %f\n", i, j, m(i, j));
        }
    }
    std::fprintf(stderr, "Address %p\n", &m(60000, 20));
    std::fprintf(stderr, "Address %p\n", &m(4000, 20));
    m(3999, 11) = 0;
    std::fprintf(stderr, "Rows: %zu. Columns: %zu. Capacity: %zu. Nonzeros: %zu\n",
                 m.rows(), m.columns(), m.capacity(), m.nonZeros());
    auto zomg = m * n;
    for(auto i(0l); i < (long)zomg.rows(); ++i) {
        for(auto j(0l); j < (long)zomg.columns(); ++j) {
            std::fprintf(stderr, "zomg %ld, %ld is %f\n", i, j, zomg(i, j));
        }
    }
#endif
}
