#ifndef _LINEAR_SVM_H_
#define _LINEAR_SVM_H_
#include "lib/misc.h"
#include "lib/parse.h"
#include "lib/metakernel.h"
#include "lib/matrixkernel.h"
#include "lib/mathutil.h"
#include "lib/learning_rate.h"
#include "lib/loss.h"

namespace svm {

template<typename FloatType, typename WeightMatrixKind=DynamicMatrix<FloatType>>
class WeightMatrix {
    FloatType norm_;
public:
    WeightMatrixKind weights_;
    operator WeightMatrixKind&() {return weights_;}
    operator const WeightMatrixKind&() const {return weights_;}
    WeightMatrix(size_t ns, size_t nc, FloatType lambda):
        norm_{0.}, weights_{WeightMatrixKind(nc == 2 ? 1: nc, ns)} {
    }
    WeightMatrix(): norm_(0.) {}
    void scale(FloatType factor) {
        norm_ *= factor * factor;
        weights_ *= factor;
    }
    FloatType get_norm_sq() {
        return norm_ = ::blaze::norm(row(weights_, 0));
    }
    FloatType norm() const {return norm_;}
};

template<typename FloatType=FLOAT_TYPE,
         class MatrixKind=DynamicMatrix<FloatType>,
         class LearningPolicy=PegasosLearningRate<FloatType>,
         class LossFn=LossSubgradient<FloatType, HingeSubgradientCore<FloatType>>,
         typename LabelType=std::int16_t>
class StreamingSVM {

    // Increase nd by 1 and set all the last entries to "1" to add
    // the bias term implicitly.
    // (See http://ttic.uchicago.edu/~nati/Publications/PegasosMPB.pdf,
    //  section 6.)
    // This will require more rewriting than the other SVM implementations to support multiclass classification.

    using WMType       = DynamicMatrix<FloatType>;
    using KernelType   = LinearKernel<FloatType>;

    WMType                    w_;
    WMType                w_avg_;
    const FloatType      lambda_; // Lambda Parameter
    const KernelType     kernel_;
    const LossFn       &loss_fn_;
    static constexpr size_t nc_ = 2;
    const size_t            mbs_; // Mini-batch size
    size_t                   nd_; // Number of dimensions
    const size_t       max_iter_; // Maximum iterations.
                                  // If -1, use epsilon termination conditions.
    size_t                    t_; // Timepoint.
    const LearningPolicy     lp_; // Calculates learning rate at a timestep t.
    const FloatType         eps_; // epsilon termination.
    const size_t       avg_size_; // Number to average at end.
    const bool          project_; // Whether or not to perform projection step.
    const bool            scale_; // Whether or not to scale to unit variance and 0 mean.
    const bool             bias_; // Whether or not to add an additional dimension to account for bias.

public:
    // Dense constructor
    StreamingSVM(const FloatType lambda, size_t ndims,
                 LearningPolicy lp,
                 size_t mini_batch_size=256uL,
                 size_t max_iter=100000,  const FloatType eps=1e-6,
                 long avg_size=-1, bool project=true, bool scale=false, bool bias=true, const LossFn &fn={})
        : lambda_(lambda),
          nc_(2), mbs_(mini_batch_size), nd_(ndims + static_cast<int>(bias)),
          max_iter_(max_iter), t_(0), lp_(lp), eps_(eps < 0 ? -std::numeric_limits<FloatType>::infinity(): eps),
          avg_size_(avg_size < 0 ? 1000: avg_size),
          project_(project), scale_(scale), bias_(bias), loss_fn_{fn}
    {
    }
    size_t get_ndims()    const {return nd_;}
    bool get_bias()       const {return bias_;}
    auto  &get_matrix()         {return m_;}


public:
    // Only use the "predict" functions if you know what you're doing!
    template<typename RowType>
    FloatType predict(const RowType &datapoint) const {
        return kernel_(row(w_.weights_, 0), datapoint);
    }

    FloatType predict(const FloatType *datapoint) const {
#if 0
        const FloatType ret(blas_dot(nd_, datapoint, 1, &w_.weights_(0, 0), 1));
        FloatType tmp(0.);
        auto wr(row(w_.weights_, 0));
        for(size_t i(0); i < nd_; ++i) tmp += datapoint[i] * w_[i];
        assert(tmp == ret);
#endif
        return kernel_(row(w_.weights_, 0), datapoint);
    }

    template<typename RowType>
    int classify_external(RowType &data) const {
        return classify(data); // Left for interface compatibility with other SVMs in stochasticSVM.
    }
    template<typename RowType>
    int classify(const RowType &data) const {
        static const int tbl[]{-1, 1};
        return tbl[predict(data) > 0.];
    }
    void train() {
#if !NDEBUG
        const size_t interval(max_iter_ / 10);
#endif
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
    void cleanup() {
        free_matrix(m_);
        free_vector(v_);
        row(w_.weights_, 0) = row(w_avg_.weights_, 0);
        free_matrix(w_avg_.weights_);
    }
    void write(FILE *fp, bool scientific_notation=false) {
        fprintf(fp, "#Dimensions: %zu.\n", nd_);
        fprintf(fp, "#LinearKernel\n");
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
    auto eps()       const {return eps_;}
    auto lambda()    const {return lambda_;}
    auto max_iter()  const {return max_iter_;}
    auto t()         const {return t_;}

    template<typename... Args>
    decltype(auto) kernel(Args &&... args)    const {return kernel_(std::forward(args)...);}

    // Reference getters
    auto &w()              {return w_;}
    const auto &v()  const {return v_;}
    const auto &m()  const {return m_;}
    const auto &lp() const {return lp_;}
}; // LinearSVM

} // namespace svm
#if !NDEBUG
#  undef NOTIFICATION_INTERVAL
#endif


#endif // _LINEAR_SVM_H_
