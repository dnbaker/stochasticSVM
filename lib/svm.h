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

template<typename MatrixType, class MatrixKind>
class WeightMatrix {
    MatrixType norm_;
    MatrixKind weights_;
public:
    operator MatrixKind&() {return weights_;}
    operator const MatrixKind&() const {return weights_;}
    WeightMatrix(size_t ns, size_t nc, MatrixType lambda):
        norm_{0.}, weights_{MatrixKind(ns, nc == 2 ? 1: nc)} {
        const MatrixType val(ns == 2 ? std::sqrt(1 / lambda) / ns: 0.);
        for(size_t i(0); i < weights_.rows(); ++i)
            for(size_t j(0); j < weights_.rows(); ++j)
                weights_(i, j) = val;
    }
    WeightMatrix(): norm_(0.) {}
    void scale(MatrixType factor) {
        norm_ *= factor * factor;
        weights_ *= factor;
    }
};

template<class Kernel,
         typename MatrixType=float,
         class MatrixKind=DynamicMatrix<MatrixType>,
         typename VectorType=int,
         class LearningPolicy=PegasosLearningRate<MatrixType>>
class SVM {

    using WMType = WeightMatrix<MatrixType, MatrixKind>;

    MatrixKind                m_; // Training Data
    // Weights. one-dimensional for 2-class, nc_-dimensional for more.
    WMType w_;
    WMType w_avg_;
    DynamicVector<VectorType> a_; // Only used for kernel.
    DynamicVector<VectorType> v_; // Labels
    MatrixKind r_; // Renormalization values. Subtraction, then multiplication
    const MatrixType     lambda_; // Lambda Parameter
    size_t                   nc_; // Number of classes
    size_t                  mbs_; // Mini-batch size
    size_t                   ns_; // Number samples
    size_t                   nd_; // Number of dimensions
    size_t             max_iter_; // Maximum iterations.
                                  // If -1, use epsilon termination conditions.
    size_t                    t_; // Timepoint.
    LearningPolicy           lp_; // Calculates learning rate at a timestep t.
    MatrixType              eps_; // epsilon termination.
    std::unordered_map<VectorType, std::string> class_name_map_;
    

public:
    SVM(const char *path, const MatrixType lambda, size_t mini_batch_size,
        size_t max_iter=1000000,  const MatrixType eps=1e-12)
        : lambda_(lambda), nc_(0), mbs_(mini_batch_size), t_(0),
          max_iter_(max_iter), lp_(lambda), eps_(eps) {
        load_data(path);
    }
    size_t get_nsamples() {return ns_;}
    size_t get_ndims()    {return nd_;}
    auto  &get_matrix()   {return m_;}


private:
    void load_data(const char *path) {
        dims_t dims(path);
        ns_ = dims.ns_; nd_ = dims.nd_;
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
        w_ = WMType(ns_, nc_ == 2 ? 1: nc_, lambda_);
        rescale();
    }
    void rescale() {
        r_ = MatrixKind(nd_, 2);
        // Could/Should rewrite with pthread-type parallelization and get better memory access pattern.
        #pragma omp parallel for schedule(dynamic)
        for(size_t i = 0; i < nd_; ++i) {
            auto col(column(m_, i));
            assert(ns_ == col.size());
            auto sum(0.);
            for(auto c: col) sum += c;
            MatrixType mean = sum / ns_;
            r_(i, 0) = mean;
            auto var(variance(col, mean));
            MatrixType stdev_inv;
            r_(i, 1) = stdev_inv = 1. / (MatrixType)std::sqrt(var);
            for(auto cit(col.begin()), cend(col.end()); cit != cend; ++cit) {
                *cit = (*cit - mean) * r_(i, 1);
            }
            var = variance(col, 0.);
        }
    }
    void train_linear() {
        while(t_ < max_iter_) {
            blaze::Subvector labels(v_, t_ * mbs_, mbs_);
        }
    }
    // Training
    // For kernel, see fig. 3. http://ttic.uchicago.edu/~nati/Publications/PegasosMPB.pdf
    // For linear, see section 2. http://www.ee.oulu.fi/research/imag/courses/Vedaldi/ShalevSiSr07.pdf
    // Consists of a gradient step ∇t = λ wt - \frac{1}{|A_t|}\sum{x,y \in A_{minibatch}}y^Tx
    // followed by a projection onto a ball around w_t.
    // I think it should be simple enough to write one that handles both linear and kernels.
}; // SVM

} // namespace svm


#endif // _PROBLEM_H_
