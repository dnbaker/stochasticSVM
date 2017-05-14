#include "lib/kernel_svm.h"
#include "src/run_svm.h"

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
                       "-[h?]:\tHelp menu.\n"\
                       "-l:\tSet lambda parameter.\n"\
                       KERNEL_USAGE \
                       "-M:\tMax iter (100000)\n"\
                       "-b:\tBatch size\n"\
                       "-e:\tSet epsilon for termination.\n"\
                       "-s:\tNumber of dimensions for sparse parsing. Also determines the use of sparse rather than dense parsing.\n"\
                 , ex);\
    cerr << buf;\
    return EXIT_FAILURE;\
}\
\
int main(int argc, char *argv[]) {\
    int c, batch_size(256), nd_sparse(0);\
    FLOAT_TYPE lambda(0.5), eps(1e-6);\
    KERNEL_PARAMS\
    size_t max_iter(100000);\
    unsigned nthreads(1);\
    FILE *ofp(stdout);\
    bool rescale(false);\
    bool bias(true);\
    while((c = getopt(argc, argv, KERNEL_GETOPT "e:M:s:p:b:l:o:Brh?")) >= 0) {\
        switch(c) {\
            case 'B': bias       = false;        break;\
            case 'e': eps        = atof(optarg); break;\
            case 'p': nthreads   = atoi(optarg); break;\
            case 'M': max_iter   = strtoull(optarg, 0, 10); break;\
            case 'b': batch_size = atoi(optarg); break;\
            case 'l': lambda     = atof(optarg); break;\
            case 's': nd_sparse  = atoi(optarg); break;\
            case 'r': rescale    = true; break;\
            KERNEL_ARGS\
            case 'o': ofp        = fopen(optarg, "w"); break;\
                if(ofp == nullptr) throw std::runtime_error(\
                    std::string("Could not open file at ") + optarg);\
                break;\
            case 'h': case '?': usage: return usage(*argv);\
        }\
    }\
\
    if(optind == argc) goto usage;\
    blaze::setNumThreads(nthreads);\
    omp_set_num_threads(nthreads);\
    KERNEL_INIT;\
    cerr << "Training data at " << argv[optind] << ".\n";\
    if(optind < argc - 1) cerr << "Test data at " << argv[optind + 1] << ".\n";\
    KernelSVM<decltype(kernel), FLOAT_TYPE> svm(\
            nd_sparse ? KernelSVM<decltype(kernel), FLOAT_TYPE>(argv[optind], nd_sparse, lambda, kernel, batch_size, max_iter, eps, rescale, bias)\
                      : KernelSVM<decltype(kernel), FLOAT_TYPE>(argv[optind], lambda, kernel, batch_size, max_iter, eps, rescale, bias));\
    RUN_SVM\
    if(ofp != stdout) fclose(ofp);\
}
