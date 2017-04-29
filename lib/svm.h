#ifndef _PROBLEM_H_
#define _PROBLEM_H_
#include "lib/misc.h"
#include "blaze/Math.h"
#include "lib/parse.h"
#include "lib/kernel.h"
#include "lib/mkernel.h"
#include "lib/math.h"
#include "lib/learning_rate.h"
#include "fastrange/fastrange.h"

namespace svm {


// TODONE: Polynomial kernels
// TODONE: Gradients

template<typename MatrixType, typename WeightMatrixKind=CompressedMatrix<MatrixType>>
class WeightMatrix {
    MatrixType norm_;
public:
    WeightMatrixKind weights_;
    operator WeightMatrixKind&() {return weights_;}
    operator const WeightMatrixKind&() const {return weights_;}
    WeightMatrix(size_t ns, size_t nc, MatrixType lambda):
        norm_{0.}, weights_{WeightMatrixKind(nc == 2 ? 1: nc, ns)} {}
    WeightMatrix(): norm_(0.) {}
    void scale(MatrixType factor) {
        norm_ *= factor * factor;
        weights_ *= factor;
    }
    MatrixType get_norm_sq() {
        LOG_DEBUG("Trying to make a norm_sq\n");
        try {
            LOG_DEBUG("If you see this, then we didn't do the exception\n");
            return norm_ = dot(row(weights_, 0), row(weights_, 0));
        } catch(std::invalid_argument &ia) {
            LOG_DEBUG("Dot product between row of size %zu and %zu failed...\n", row(weights_, 0).size(), row(weights_, 0).size());
            MatrixType ret(0.);
            const auto wrow(row(weights_, 0));
            for(const auto i: wrow) {
                ret += i.value() * i.value();
            }
            return ret;
        }
    }
};

template<class Kernel, typename MatrixType=float,
         class MatrixKind=DynamicMatrix<MatrixType>,                            
         typename VectorType=int>
class SVMClassifier {
    // TODO: Add bias parameter.
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
        } else {
            throw std::runtime_error("NotImplementedError");
        }
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

    using WMType = WeightMatrix<MatrixType>;

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
        size_t mini_batch_size=1,
        size_t max_iter=1000,  const MatrixType eps=1e-12,
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
    template<typename RowType>
    double predict_linear(RowType &datapoint) {
        double ret(0.);
        LOG_DEBUG("About to start predicting. Number of rows in weights: %zu\n", w_.weights_.rows());
        // Nota bene: the CompressedMatrix type skips all zero entries, so classification is pretty fast while sparse.
//#if USE_COMPRESSED
#if 1
        // This can be done with std::enable_if, but I'm lazy for now
        for(auto i(w_.weights_.cbegin(0)), e(w_.weights_.cend(0));
            i != e; ++i) {
            LOG_DEBUG("Size of row: %zu. dp: %zu\n", row(m_, i->index()).size(), datapoint.size());
            ret += i->value() * kernel_(row(m_, i->index()), datapoint);
        }
#else
        for(auto i(0ul); i < ns_; ++i) {
            ret += kernel_(row(m_, i), datapoint);
        }
#endif
        LOG_DEBUG("Okay I'm done. Prediction: %lf\n", ret);
        return ret;
    }
    template<typename WeightMatrixKind>
    void add_entry_linear(const size_t index, WeightMatrixKind &tmpsum, size_t &nels_added) {
        LOG_DEBUG("Get mrow and wrow for index: %zu\n", index);
        auto mrow(row(m_, index));
        if(predict_linear(mrow) * v_[index] < 1.) {
            LOG_DEBUG("LOSS! Size of row: %zu. Size of matrix row: %zu\n", row(tmpsum, 0).size(), mrow.size());
            tmpsum(0, index) += v_[index];
        } else {
            LOG_DEBUG("No loss!\n");
        }
        LOG_DEBUG("kernel evaluated\n");
        ++nels_added;
    }
    template<typename WeightMatrixKind>
    void add_block_linear(const size_t index, WeightMatrixKind &tmpsum,
                          size_t &nels_added) {
        for(size_t i(index), end(index + mbs_); i < end; ++i) {
            //cerr << "Adding block at index " << i << ", which is less than " << end << ".\n";
            assert(index < end);
            add_entry_linear(index, tmpsum, nels_added);
        }
        LOG_DEBUG("All blocks added for block starting at %zu\n", index);
    }
public:
    void train_linear() {
        size_t avgs_used(0);
        decltype(w_.weights_) tmpsum(1, ns_);
        size_t nels_added(0);
        LOG_DEBUG("About to start training\n");
        auto wrow(row(w_.weights_, 0));
        auto trow(row(tmpsum, 0));
        LOG_DEBUG("Set wrow and trow\n");
        for(t_ = 0; avgs_used < avg_size_; ++t_) {
            if((t_ & 255uL) == 0) {
                cerr << "Weights currently: " << w_.weights_ << '\n';
            }
            const double eta(lp_(t_));
            const size_t start_index = rand64() % std::min(ns_ - mbs_, ns_);
            LOG_DEBUG("Start index: %zu\n", start_index);
            for(auto &i: row(tmpsum, 0)) i = 0.;
            add_block_linear(start_index, tmpsum, nels_added);
            w_.scale(1.0 - eta * lambda_);
            const double scale_factor = eta / nels_added;
            LOG_DEBUG("About to add tmp row to weight row\n");
            wrow += trow * scale_factor;
            LOG_DEBUG("About to get row norm\n");
            const double norm(w_.get_norm_sq());
            if(norm > 1. / lambda_) {
                LOG_DEBUG("Scaling down bc too big\n");
                w_.scale(std::sqrt(1.0 / (lambda_ * norm)));
            }
            if(t_ >= max_iter_ || false) { // TODO: replace false with epsilon
                if(w_avg_.weights_.rows() == 0) w_avg_ = WMType(ns_, nc_ == 2 ? 1: nc_, lambda_);
                auto avg_row(row(w_avg_.weights_, 0));
                LOG_DEBUG("Updating averages\n");
                avg_row += wrow;
                ++avgs_used;
            }
            LOG_DEBUG("Finishing loop\n");
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
