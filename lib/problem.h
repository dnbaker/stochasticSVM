#ifndef _PROBLEM_H_
#define _PROBLEM_H_
#include "lib/misc.h"
#include "blaze/Math.h"
#include "lib/parse.h"
#include "lib/kernel.h"
#include "lib/mkernel.h"
#include "lib/learning_rate.h"

namespace svm {


// TODO: Polynomial kernels
// TODO: Gradients


template<class Kernel, typename MatrixType=float, typename VectorType=int, class LearningPolicy=PegasosLearningRate<MatrixType>>
class SVM {
    DynamicMatrix<MatrixType, blaze::rowMajor> m_; // Training Data
    DynamicMatrix<MatrixType> w_; // Only used in classification for kernel, though used for traiing and classification in linear.
    DynamicVector<VectorType> a_; // Only used for kernel.
    DynamicVector<VectorType> v_; // Labels
    DynamicMatrix<VectorType> r_; // Renormalization values. Subtraction, then multiplication
    const MatrixType     lambda_; // Lambda Parameter
    size_t                   nc_; // Number of classes
    size_t                  mbs_; // Mini-batch size
    size_t                   ns_; // Number samples
    size_t                   nd_; // Number of dimensions
    LearningPolicy           lp_; // Calculates learing rate at a timestep t.
    std::unordered_map<VectorType, std::string> class_name_map_;
    

public:
    SVM(const char *path, const MatrixType lambda, size_t mini_batch_size)
        : lambda_(lambda), nc_(0), mbs_(mini_batch_size), lp_(lambda_) {
        load_data(path);
    }
    size_t get_nsamples() {return ns_;}
    size_t get_ndims()    {return nd_;}
    auto  &get_matrix()   {return m_;}


private:
    void load_data(const char *path) {
        dims_t dims(path);
        ns_ = dims.ns_, nd_ = dims.nd_;
        std::tie(m_, v_, class_name_map_) = parse_problem<MatrixType, VectorType>(path, dims);
        // Normalize v_
        std::set<VectorType> set;
        for(auto &pair: class_name_map_) set.insert(pair.first);
        std::vector<VectorType> vec(std::begin(set), std::end(set));
        std::sort(std::begin(vec), std::end(vec));
        std::unordered_map<VectorType, int> map;
        int index(0);
        if(vec.size() == 2) map[vec[0]] = -1, map[vec[1]] = 1;
        else for(auto i(std::begin(vec)), e(std::end(vec)); i != e; ++i) map[*i] = ++index;
        for(auto &i: v_) i = map[i];
        decltype(class_name_map_) new_cmap;
        for(auto &pair: class_name_map_) new_cmap[map[pair.first]] = pair.second;
        class_name_map_ = std::move(new_cmap);
        nc_ = map.size();
        //init_weights();
        w_ = DynamicMatrix<MatrixType>(ns_, nc_ == 2 ? 1: nc_);
        rescale();
    }
    void rescale() {
        r_ = DynamicMatrix<MatrixType>(nd_, 2);
        // Could/Should rewrite with pthread-type parallelization and get better memory access pattern.
        #pragma omp parallel for schedule(dynamic)
        for(size_t i = 0; i < nd_; ++i) {
            MatrixType min(std::numeric_limits<MatrixType>::max()), max(std::numeric_limits<MatrixType>::min());
            for(size_t j = 0; j < ns_; ++j) {
                if(m_(i, j) > max) max = m_(i, j);
                if(m_(i, j) < min) min = m_(i, j);
            }
            r_(i, 0) = min;
            const MatrixType factor(1. / (max - r_(i, 0)));
            r_(i, 1) = factor;
            for(auto &c: column(m_, i)) c = (c - min) * factor;
        }
    }
    // If linear, initialize w_ to any with norm \geq 1/lambda.
#if 0
    template<typename = std::enable_if<std::is_same<LinearKernel<double>, Kernel>::value ||
                                       std::is_same<LinearKernel<float>, Kernel>::value>
    void init_weights() {
        w_ = std::sqrt(1. / lambda_) / ns_;
    }
    // Otherwise, initialize to 0. This instead holds the number of times a nonzero loss was found with this element.
    template<typename = std::enable_if<!std::is_same<LinearKernel<double>, Kernel>::value &&
                                       !std::is_same<LinearKernel<float>, Kernel>::value>
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
