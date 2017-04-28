#ifndef _PROBLEM_H_
#define _PROBLEM_H_
#include "lib/misc.h"
#include "blaze/Math.h"
#include "lib/parse.h"
#include "lib/kernel.h"
#include "lib/mkernel.h"
#include "lib/learning_rate.h"
#include "fastrange/fastrange.h"

namespace svm {


// TODONE: Polynomial kernels
// TODONE: Gradients

template<typename MatrixType, class MatrixKind>
class WeightMatrix {
    MatrixType norm_;
public:
    MatrixKind weights_;
    operator MatrixKind&() {return weights_;}
    operator const MatrixKind&() const {return weights_;}
    WeightMatrix(size_t ns, size_t nc, MatrixType lambda):
        norm_{0.}, weights_{MatrixKind(nc == 2 ? 1: nc, ns)} {
        const MatrixType val(ns == 2 ? std::sqrt(1 / lambda) / ns: 0.);
        weights_ = val;
    }
    WeightMatrix(): norm_(0.) {}
    void scale(MatrixType factor) {
        norm_ *= factor * factor;
        weights_ *= factor;
    }
    MatrixType get_norm_sq() {
        return norm_ = dot(row(weights_, 0), row(weights_, 0));
    }
};

template<class Kernel, typename MatrixType=float,
         class MatrixKind=DynamicMatrix<MatrixType>,                            
         typename VectorType=int>
class SVMClassifier {
    const Kernel         kernel_;
    size_t                   nc_; // Number of classes
    size_t                   nd_; // Number of dimensions
    size_t                 msvs_;
    size_t                 nsvs_;
    std::unordered_map<VectorType, std::string> class_name_map_;
    MatrixKind              svs_;
public:
    template<typename RowType>
    VectorType classify(RowType &data) const {
        double sum(0.);
        for(size_t i(0); i < svs_.rows(); ++i) {
            sum += kernel_(row(svs_, i), data);
        }
        if(nc_ == 2) {
            return std::signbit(sum);
        }
        else throw std::runtime_error("NotImplementedError");
    }
    template<typename ClassifyMatrixKind>
    DynamicVector<VectorType> classify(ClassifyMatrixKind &data) const {
        DynamicVector<VectorType> ret(data.rows());
        for(size_t i(0); i < data.rows(); ++i) ret[i] = classify(row(data, i));
        return ret;
    }
    template<typename RowType>
    size_t add_sv(RowType &sv) {
        if(msvs_ < svs_.rows() + 1) {
            msvs_ = msvs_ ? msvs_ << 1: 16;
            svs_.resize(msvs_, nd_);
        }
        row(svs_, nsvs_++) = sv;
        return nsvs_;
    }
    SVMClassifier(Kernel kernel, size_t nc, size_t nd, std::unordered_map<VectorType, std::string> class_name_map):
        kernel_(kernel), nc_(nc), nd_(nd), nsvs_(0), msvs_(0), class_name_map_(class_name_map) {}
};

template<class Kernel,
         typename MatrixType=float,
         class MatrixKind=DynamicMatrix<MatrixType>,
         typename VectorType=int,
         class LearningPolicy=PegasosLearningRate<MatrixType>>
class SVMTrainer {

    using WMType = WeightMatrix<MatrixType, MatrixKind>;

    MatrixKind                m_; // Training Data
    // Weights. one-dimensional for 2-class, nc_-dimensional for more.
    WMType w_;
    WMType w_avg_;
    DynamicVector<VectorType> a_; // Only used for kernel.
    DynamicVector<VectorType> v_; // Labels
    MatrixKind r_; // Renormalization values. Subtraction, then multiplication
    const MatrixType     lambda_; // Lambda Parameter
    const Kernel         kernel_;
    size_t                   nc_; // Number of classes
    size_t                  mbs_; // Mini-batch size
    size_t                   ns_; // Number samples
    size_t                   nd_; // Number of dimensions
    size_t             max_iter_; // Maximum iterations.
                                  // If -1, use epsilon termination conditions.
    size_t                    t_; // Timepoint.
    LearningPolicy           lp_; // Calculates learning rate at a timestep t.
    MatrixType              eps_; // epsilon termination.
    size_t             avg_size_; // Number to average at end.
    std::unordered_map<VectorType, std::string> class_name_map_;
    

public:
    SVMTrainer(const char *path,
        const MatrixType lambda,
        Kernel kernel=LinearKernel<MatrixType>(),
        size_t mini_batch_size=1<<8,
        size_t max_iter=1000000,  const MatrixType eps=1e-12,
        long avg_size=-1)
        : lambda_(lambda), kernel_(std::move(kernel)),
          nc_(0), mbs_(mini_batch_size),
          max_iter_(max_iter), t_(0), lp_(lambda), eps_(eps),
          avg_size_(avg_size < 0 ? 1000: avg_size) {
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
        }
    }
    void add_entry_linear(const size_t index, DynamicMatrix<MatrixType> &tmpsum, size_t &nels_added) {
        auto mrow(row(m_, index));
        if(kernel_(mrow, row(w_.weights_, 0)) * v_[index] < 0)
            row(tmpsum, 0) += mrow * v_[index];
        ++nels_added;
    }
    void add_block_linear(const size_t index, DynamicMatrix<MatrixType> &tmpsum,
                          size_t &nels_added) {
        for(size_t i(index), end(index + mbs_); i < end; ++i)
            add_entry_linear(index, tmpsum, nels_added);
    }
public:
    void train_linear() {
        const double eta(lp_(t_));
        size_t avgs_used(0);
        DynamicMatrix<MatrixType> tmpsum(1, nd_);
        size_t nels_added(0);
        while(avgs_used < avg_size_) {
            const size_t start_index = fastrangesize(rand64(), ns_ - mbs_);
            tmpsum = 0.;
            add_block_linear(start_index, tmpsum, nels_added);
            w_.scale(1.0 - eta * lambda_);
            const double scale_factor = eta / nels_added;
            row(w_.weights_, 0) += row(tmpsum, 0) * scale_factor;
            const double norm(w_.get_norm_sq());
            if(norm > 1. / lambda_) w_.scale(std::sqrt(1.0 / (lambda_ * norm)));
            if(t_ >= max_iter_ || false) { // TODO: replace false with epsilon
                if(avgs_used == 0) w_avg_.weights_ = 0;
                row(w_avg_.weights_, 0) += row(w_.weights_, 0);
                ++avgs_used;
            }
        }
        row(w_avg_.weights_, 0) *= 1. / avg_size_;
    }
    SVMTrainer<Kernel, MatrixType, VectorType> build_linear_classifier(double eps) {
        // If w_[i] >= eps, include in classifier.
        SVMClassifier ret(kernel_, nc_, nd_, class_name_map_);
        auto wrow(row(w_.weights, 0));
        for(size_t i(0); i < wrow.size(); ++i) {
            if(wrow[i] >= eps) ret.add_sv(wrow);
        }
        return ret;
    }
    // Training
    // For kernel, see fig. 3. http://ttic.uchicago.edu/~nati/Publications/PegasosMPB.pdf
    // For linear, see section 2. http://www.ee.oulu.fi/research/imag/courses/Vedaldi/ShalevSiSr07.pdf
    // Consists of a gradient step ∇t = λ wt - \frac{1}{|A_t|}\sum{x,y \in A_{minibatch}}y^Tx
    // followed by a projection onto a ball around w_t.
    // I think it should be simple enough to write one that handles both linear and kernels.
}; // SVMTrainer

} // namespace svm


#endif // _PROBLEM_H_
