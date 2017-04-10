#ifndef _PROBLEM_H_
#define _PROBLEM_H_
#include "lib/misc.h"
#include "blaze/Math.h"
#include "lib/kernel.h"
#include "lib/mkernel.h"
#include "lib/learning_rate.h"

namespace svm {


// TODO: Polynomial kernels
// TODO: Gradients


template<class Kernel, typename MatrixType=float, typename VectorType=int, class LearningPolicy=PegasosLearningRate<MatrixType>>
class SVM {
    DynamicMatrix<MatrixType> m_; // Training Data
    DynamicMatrix<MatrixType> w_; // Only used in classification for kernel, though used for traiing and classification in linear.
    DynamicVector<VectorType> a_; // Only used for kernel.
    DynamicVector<VectorType> v_; // Labels
    const MatrixType     lambda_; // Lambda Parameter
    size_t                   nc_; // Number of classes
    size_t                  mbs_; // Mini-batch size
    size_t                   ns_; // Number samples
    size_t                   nd_; // Number of dimensions
    LearningPolicy           lp_; // Calculates learing rate at a timestep t.
    

public:
    SVM(const char *path, MatrixType lambda, size_t mini_batch_size)
        : lambda_(lambda), nc_(0), mbs_(mini_batch_size), lp_(lambda_) {
        load_data(path);
    }
    size_t get_nsamples() {return ns_;}
    size_t get_ndims()    {return nd_;}


private:
    void load_data(const char *path) {
        std::ios::sync_with_stdio(false);
        std::pair<size_t, unsigned> dims(count_dims(path));
        nd_ = dims.second;
        ns_ = dims.first;
        m_ = DynamicMatrix<MatrixType>(ns_, nd_);
        v_ = DynamicVector<VectorType>(ns_);
        gzFile fp(gzopen(path, "rb"));
        if(fp == nullptr) throw std::runtime_error(std::string("Could not open file at ") + path);
        std::string line;
        size_t linenum(0);
        char *p;
        int c;
        int max(16), *t((int *)malloc(max * sizeof(int)));
        while((c = gzgetc(fp)) != EOF) {
            line += c;
            if(c != '\n') continue;
            // Makeshift buffer with std::string.
            p = &line[0];
            //const auto pc(line.data());
            const int ntoks(ksplit_core(p, 0, &max, &t));
            if(nd_ != ntoks - 1) LOG_EXIT("Expected %i data rows. Found %i\n", nd_, ntoks - 1);
            for(int i(0); i < ntoks - 1; ++i) m_(linenum, i) = atof(p + t[i]);
            v_[linenum++] = atoi(p + t[ntoks - 1]);
            if(linenum % 10000 == 0) std::fprintf(stderr, "Parsed %zu lines\n", linenum);
            line.clear();
        }
        free(t);
        gzclose(fp);
        std::fprintf(stderr, "linenum: %zu. num rows: %zu. cols: %zu.\n", linenum, m_.rows(), m_.columns());
        assert(linenum == ns_);
        std::set<VectorType> set(std::begin(v_), std::end(v_));
        std::vector<VectorType> vec(std::begin(set), std::end(set));
        std::sort(std::begin(vec), std::end(vec));
        std::map<VectorType, int> map;
        int index(0);
        if(vec.size() == 2) {
            map[vec[0]] = -1;
            map[vec[1]] = 1;
        } else {
            for(auto i(std::begin(vec)), e(std::end(vec)); i != e; ++i) map[*i] = ++index;
        }
        for(auto &i: v_) i = map[i];
        nc_ = map.size();
        //init_weights();
        w_ = DynamicMatrix<MatrixType>(ns_, nc_ == 2 ? 1: nc_);
    }
#if 0
    // If linear, initialize w_ to any with norm \geq 1/lambda.
    template<typename = std::enable_if<std::is_same<LinearKernel, Kernel>::value>>
    void init_weights() {
        w_ = std::sqrt(1. / lambda_) / ns_;
    }
    // Otherwise, initialize to 0. This instead holds the number of times a nonzero loss was found with this element.
    template<typename = std::enable_if<!std::is_same<LinearKernel, Kernel>::value>>
    void init_weights() {w_ = 0;}
#endif
    // Training
    // For kernel, see fig. 3. http://ttic.uchicago.edu/~nati/Publications/PegasosMPB.pdf
    // For linear, see section 2. http://www.ee.oulu.fi/research/imag/courses/Vedaldi/ShalevSiSr07.pdf
    // Consists of a gradient step ∇t = λ wt - \frac{1}{|A_t|}\sum{x,y \in A_{minibatch}}y^Tx
    // followed by a projection onto a ball around w_t.
    // I think it should be simple enough to write one that handles both linear and kernels.
}; // SVM

} // namespace svm


#endif // _PROBLEM_H_
