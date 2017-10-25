#include "lib/linear_svm.h"
#include "src/train_svm.h"


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
                       "-r: Rescale data. Default: false.\n"
                 , ex);
    cerr << buf;
    return EXIT_FAILURE;
}

enum Policy:size_t{
    PEGASOS = 0,
    NORMA   = 1,
    FIXED   = 2
};

#define TRAIN_SVM(policy, MatrixType) \
        LinearSVM<FLOAT_TYPE, MatrixType<FLOAT_TYPE>, decltype(policy)> svm = \
            nd_sparse ? LinearSVM<FLOAT_TYPE, MatrixType<FLOAT_TYPE>, decltype(policy)>(argv[optind], nd_sparse, lambda, policy, batch_size, max_iter, eps, 1000, project, rescale, bias) \
                      : LinearSVM<FLOAT_TYPE, MatrixType<FLOAT_TYPE>, decltype(policy)>(argv[optind], lambda, policy, batch_size, max_iter, eps, 1000, project, rescale, bias)

#define TRAIN_SPARSE_SVM(policy) TRAIN_SVM(policy, CompressedMatrix)
#define TRAIN_DENSE_SVM(policy)  TRAIN_SVM(policy, DynamicMatrix)

int main(int argc, char *argv[]) {
    int c, batch_size(256), nd_sparse(0);
    FLOAT_TYPE lambda(0.5), eta(0.0), eps(1e-6);
    size_t max_iter(100000);
    unsigned nthreads(1);
    std::ios::sync_with_stdio(false);
    FILE *ofp(stdout);
    Policy policy(PEGASOS);
    bool project(false);
    bool rescale(false);
    bool use_sparse(false);
    bool bias(true);
    for(char **p(argv + 1); *p; ++p) if(strcmp(*p, "--help") == 0) goto usage;
    while((c = getopt(argc, argv, "E:e:M:s:P:p:b:l:o:BrFNh?")) >= 0) {
        switch(c) {
            case '5': use_sparse = true;         break;
            case 'B': bias       = false;        break;
            case 'E': eta        = atof(optarg); break;
            case 'e': eps        = atof(optarg); break;
            case 'p': nthreads   = atoi(optarg); break;
            case 'P': project    = true;         break;
            case 'M': max_iter   = strtoull(optarg, 0, 10); break;
            case 'b': batch_size = atoi(optarg); break;
            case 'l': lambda     = atof(optarg); break;
            case 's': nd_sparse  = atoi(optarg); break;
            case 'N': policy     = NORMA;        break; // Guerra, Guerra!
            case 'F': policy     = FIXED;        break;
            case 'r': rescale    = true;         break;
            case 'o': if((ofp     = fopen(optarg, "w")) == nullptr)
                          throw std::runtime_error(
                              std::string("Could not open file at ") + optarg);
                                                 break;
            case 'h': case '?': usage: return usage(*argv);
        }
    }
    if(policy == NORMA && eta == 0.0) {
            eta = 1./lambda;
            cerr << "Eta unset for Norma method. Defaulting to 1/lambda: "
                 << eta << ".\n";
    }
    else if(policy == FIXED && eta == 0.0) {
        throw std::runtime_error(
            "eta (-E) must be set for Fixed Learning rate policies.");
    }

    if(optind == argc) goto usage;
    blaze::setNumThreads(nthreads);
    PegasosLearningRate<FLOAT_TYPE> plp(lambda);
    NormaLearningRate<FLOAT_TYPE>   nlp(eta);
    FixedLearningRate<FLOAT_TYPE>   flp(eta);
    if(policy == NORMA) {
        if(use_sparse) {
            TRAIN_SPARSE_SVM(nlp);
            RUN_SVM
        } else {
            TRAIN_DENSE_SVM(nlp);
            RUN_SVM
        }
    } else if(policy == FIXED) {
        if(use_sparse) {
            TRAIN_SPARSE_SVM(flp);
            RUN_SVM
        } else {
            TRAIN_DENSE_SVM(flp);
            RUN_SVM
        }
    } else {
        if(use_sparse) {
            TRAIN_SPARSE_SVM(plp);
            RUN_SVM
        } else {
            TRAIN_DENSE_SVM(plp);
            RUN_SVM
        }
    }
    if(ofp != stdout) fclose(ofp);
}
