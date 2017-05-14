#include "lib/kernel_svm.h"
#include "src/run_svm.h"
using namespace svm;

int usage(char *ex) {
    char buf[1024];
    std::sprintf(buf, "Usage: %s <opts> data\n"
                       "Flags:\n-p:\tNumber of processes [1]\n\n"
                       "-[h?]:\tHelp menu.\n"
                       "-l:\tSet lambda parameter.\n"
                       "-g:\tSet gamma parameter for RBF kernel [1.0].\n"
                       "-M:\tMax iter (100000)\n"
                       "-b:\tBatch size\n"
                       "-e:\tSet epsilon for termination.\n"
                       "-s:\tNumber of dimensions for sparse parsing. Also determines the use of sparse rather than dense parsing.\n"
                 , ex);
    cerr << buf;
    return EXIT_FAILURE;
}

class IntCounter {
    std::map<int, int> map_;
public:
    void add(int val) {
        ++map_[val];
    }
    std::string str() const {
        std::string ret("{");
        for(auto &pair: map_) ret += std::to_string(pair.first) + ": " + std::to_string(pair.second) + ", ";
        ret.pop_back();
        ret[ret.size() - 1] = '}';
        return ret;
    }
};


int main(int argc, char *argv[]) {
    int c, batch_size(256), nd_sparse(0);
    FLOAT_TYPE lambda(0.5), gamma(1.0), eps(1e-6);
    size_t max_iter(100000);
    unsigned nthreads(1);
    std::ios::sync_with_stdio(false);
    FILE *ofp(stdout);
    bool rescale(false);
    bool bias(true);
    for(char **p(argv + 1); *p; ++p) if(strcmp(*p, "--help") == 0) goto usage;
    while((c = getopt(argc, argv, "g:e:M:s:p:b:l:o:Brh?")) >= 0) {
        switch(c) {
            case 'B': bias       = false;        break;
            case 'e': eps        = atof(optarg); break;
            case 'g': gamma      = atof(optarg); break;
            case 'p': nthreads   = atoi(optarg); break;
            case 'M': max_iter   = strtoull(optarg, 0, 10); break;
            case 'b': batch_size = atoi(optarg); break;
            case 'l': lambda     = atof(optarg); break;
            case 's': nd_sparse  = atoi(optarg); break;
            case 'o': ofp        = fopen(optarg, "w"); break;
            case 'r': rescale    = true; break;
                if(ofp == nullptr) throw std::runtime_error(
                    std::string("Could not open file at ") + optarg);
                break;
            case 'h': case '?': usage: return usage(*argv);
        }
    }

    if(optind == argc) goto usage;
    blaze::setNumThreads(nthreads);
    omp_set_num_threads(nthreads);
    RBFKernel<FLOAT_TYPE> kernel(gamma);
    KernelSVM<decltype(kernel), FLOAT_TYPE> svm(
            nd_sparse ? KernelSVM<decltype(kernel), FLOAT_TYPE>(argv[optind], nd_sparse, lambda, kernel, batch_size, max_iter, eps, rescale, bias)
                      : KernelSVM<decltype(kernel), FLOAT_TYPE>(argv[optind], lambda, kernel, batch_size, max_iter, eps, rescale, bias));
    svm.train();
    svm.write(ofp);
    RUN_SVM
    if(ofp != stdout) fclose(ofp);
}
