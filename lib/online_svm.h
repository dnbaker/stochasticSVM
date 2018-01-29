#ifndef _ONLINE_SVM_H_
#define _ONLINE_SVM_H_
#include "lib/linear_svm.h"

namespace svm {

template<typename FloatType=FLOAT_TYPE,
         class MatrixKind=DynamicMatrix<FloatType>,
         class LearningPolicy=PegasosLearningRate<FloatType>,
         class LossFn=LossSubgradient<FloatType, HingeSubgradientCore<FloatType>>,
         typename LabelType=std::int16_t>
class OnlineSVM {

    // Increase nd by 1 and set all the last entries to "1" to add
    // the bias term implicitly.
    // (See http://ttic.uchicago.edu/~nati/Publications/PegasosMPB.pdf,
    //  section 6.)

    using WMType       = WeightMatrix<FloatType, DynamicMatrix<FloatType>>;
    using KernelType   = LinearKernel<FloatType>;

    // Weights. one-dimensional for 2-class, nc_-dimensional for more.
    WMType                    w_;
    WMType                w_avg_;
    // Labels [could be compressed by requiring sorted data and checking
    // an index is before or after a threshold. Likely unnecessary, but could
    // aid cache efficiency.
    const FloatType      lambda_; // Lambda Parameter
    const KernelType     kernel_;
    const LossFn       &loss_fn_;
    size_t                   nc_;
    size_t                  mbs_;
    size_t                   nd_; // Number of dimensions
    const size_t       max_iter_; // Maximum iterations.
                                  // If -1, use epsilon termination conditions.
    size_t                    t_; // Timepoint.
    const LearningPolicy     lp_; // Calculates learning rate at a timestep t.
    const size_t       avg_size_; // Number to average at end.
    const bool          project_; // Whether or not to perform projection step.
    const bool            scale_; // Whether or not to scale to unit variance and 0 mean.
    const bool             bias_; // Whether or not to add an additional dimension to account for bias.

public:
    OnlineSVM(const FloatType lambda,
              LearningPolicy lp,
              size_t nd,
              size_t mini_batch_size=256uL,
              size_t max_iter=100000,  const FloatType eps=1e-6,
              long avg_size=-1, bool project=false, bool scale=false, bool bias=true, const LossFn &fn={})
        : w_(nd + static_cast<int>(bias), 2, lambda),
          w_avg_(nd + static_cast<int>(bias), 2, lambda),
          lambda_(lambda),
          nc_(0), mbs_(mini_batch_size), nd_(nd),
          max_iter_(max_iter), t_(0), lp_(lp),
          avg_size_(avg_size < 0 ? 10: avg_size), project_(project), scale_(scale), bias_(bias), loss_fn_{fn}
    {
        if(nc != 2) throw std::runtime_error("NotImplementedError");
        
    }
    size_t get_ndims()    const {return nd_;}
    bool get_bias()       const {return bias_;}
    auto  &get_matrix()         {return m_;}


private:
    void normalize() {
        if(bias_) for(size_t i(0), e(m_.rows()); i < e; ++i) m_(i, nd_ - 1) = 1.;
    }
public:
    template<typename RowType>
    FloatType predict(const RowType &datapoint) const {
        return ::blaze::dot(row(w_.weights_, 0), datapoint);
    }

    FloatType predict(const FloatType *datapoint) const {
        return ::blaze::dot(row(w_.weights_, 0), datapoint);
    }

    template<typename RowType>
    int classify_external(RowType &data) const {
        return classify(data);
    }
    template<typename RowType>
    int classify(const RowType &data) const {
        static const int tbl[]{-1, 1};
        return tbl[predict(data) > 0.];
    }
#if 0
    void train() {
#if !NDEBUG
        const size_t interval(max_iter_ / 10);
#endif
        // Set constants
        const size_t max_end_index(std::min(mbs_, ns_));
        const FloatType batchsz_inv(1./max_end_index);

        // Allocate weight vectors and initialize row views.
        decltype(w_.weights_) tmpsum(1, nd_);
        decltype(w_.weights_) last_weights(1, nd_);
        auto wrow(row(w_.weights_, 0));
        auto trow(row(tmpsum, 0));

        // Allocate hash set
        khash_t(I) *h(kh_init(I));
        kh_resize(I, h, mbs_ * 1.5);

        //size_t avgs_used(0);
#if USE_OLD_WAY
        FloatType eta;
#endif
        for(size_t avgs_used = t_ = 0; avgs_used < avg_size_; ++t_) {
#if !NDEBUG
            if((t_ % interval) == 0) {
                const FloatType ls(loss()), current_norm(w_.get_norm_sq());
                cerr << "Loss: " << ls * 100 << "%" << " at time = " << t_ << 
                        " with norm of w = " << current_norm << ".\n";
            }
#endif
            {
                int khr;
                kh_clear(I, h);
                while(kh_size(h) < max_end_index)
                    kh_put(I, h, RANGE_SELECT(max_end_index), &khr);
            }

            // This is the part of the code that I would replace with the
            // generic projection
            loss_fn_(*this, trow, last_weights, h);
            if(project_) {
                const FloatType norm(w_.get_norm_sq());
                if(norm > 1. / lambda_)
                    w_.scale(std::sqrt(1.0 / (lambda_ * norm)));
            }
            if(t_ >= max_iter_ || (eps_ >= 0 &&
                                   diffnorm(row(last_weights, 0), row(w_.weights_, 0)) < eps_)) {
                if(w_avg_.weights_.rows() == 0)
                    w_avg_ = WMType(nd_, nc_ == 2 ? 1: nc_, lambda_), w_avg_.weights_.reset();
                row(w_avg_.weights_, 0) += wrow;
                ++avgs_used;
            }
        }
        kh_destroy(I, h);
        row(w_avg_.weights_, 0) *= 1. / avg_size_;
        FloatType ls(loss());
        cout << "Train error: " << ls * 100 << "%\nafter "
             << t_ + 1 << " iterations "<<'\n';
        cleanup();
    }
#endif
    void cleanup() {
        free_matrix(m_);
        free_vector(v_);
        row(w_.weights_, 0) = row(w_avg_.weights_, 0);
        free_matrix(w_avg_.weights_);
    }
    void write(FILE *fp, bool scientific_notation=false) {
        fprintf(fp, "#Dimensions: %zu.\n", nd_);
        fprintf(fp, "#OnlineLinearKernel\n");
        ks::KString line;
        line.resize(5 * row(w_.weights_, 0).size());
        const char *fmt(scientific_notation ? "%e, ": "%f, ");
        for(const auto i: row(w_.weights_, 0))
            line.sprintf(fmt, i);
        line.pop();
        line[line.size() - 1] = '\n';
        fwrite(line.data(), line.size(), 1, fp);
    }
    // Value getters
    auto lambda()    const {return lambda_;}
    auto max_iter()  const {return max_iter_;}
    auto t()         const {return t_;}

    // Reference getters
    auto &w()              {return w_;}
    const auto &lp() const {return lp_;}
}; // OnlineSVM

} // namespace svm
#if !NDEBUG
#  undef NOTIFICATION_INTERVAL
#endif


#endif // _ONLINE_SVM_H_
