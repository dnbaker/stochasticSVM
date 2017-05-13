#ifndef _LINEAR_SVM_H_
#define _LINEAR_SVM_H_
#include "lib/misc.h"
#include "blaze/Math.h"
#include "lib/parse.h"
#include "lib/metakernel.h"
#include "lib/matrixkernel.h"
#include "lib/mathutil.h"
#include "lib/learning_rate.h"
#include "fastrange/fastrange.h"

namespace svm {

template<typename FloatType, typename WeightMatrixKind=DynamicMatrix<FloatType>>
class WeightMatrix {
    double norm_;
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
        return norm_ = dot(row(weights_, 0), row(weights_, 0));
    }
    double norm() const {return norm_;}
};

template<typename FloatType=float,
         class MatrixKind=DynamicMatrix<FloatType>,
         class LearningPolicy=PegasosLearningRate<FloatType>>
class LinearSVM {

    // Increase nd by 1 and set all the last entries to "1" to add
    // the bias term implicitly.
    // (See http://ttic.uchicago.edu/~nati/Publications/PegasosMPB.pdf,
    //  section 6.)

    using WMType     = WeightMatrix<FloatType>;
    using KernelType = LinearKernel<FloatType>;

    MatrixKind                m_; // Training Data
    // Weights. one-dimensional for 2-class, nc_-dimensional for more.
    WMType w_;
    WMType w_avg_;
    DynamicVector<int>        v_;
    DynamicMatrix<FloatType>  r_;
    // Labels [could be compressed by requiring sorted data and checking
    // an index is before or after a threshold. Likely unnecessary, but could
    // aid cache efficiency.
    const FloatType      lambda_; // Lambda Parameter
    const KernelType     kernel_;
    size_t                   nc_; // Number of classes
    const size_t            mbs_; // Mini-batch size
    size_t                   ns_; // Number samples
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
    std::unordered_map<int, std::string> class_name_map_;

public:
    // Dense constructor
    LinearSVM(const char *path,
              const FloatType lambda,
              LearningPolicy lp,
              size_t mini_batch_size=256uL,
              size_t max_iter=100000,  const FloatType eps=1e-6,
              long avg_size=-1, bool project=true, bool scale=false, bool bias=true)
        : lambda_(lambda),
          nc_(0), mbs_(mini_batch_size),
          max_iter_(max_iter), t_(0), lp_(lp), eps_(eps < 0 ? -std::numeric_limits<float>::infinity(): eps),
          avg_size_(avg_size < 0 ? 1000: avg_size),
          project_(project), scale_(scale), bias_(bias)
    {
        load_data(path);
    }
    LinearSVM(const char *path, size_t ndims,
               const FloatType lambda,
               LearningPolicy lp,
               size_t mini_batch_size=256uL,
               size_t max_iter=100000,  const FloatType eps=1e-6,
               long avg_size=-1, bool project=false, bool scale=false, bool bias=true)
        : lambda_(lambda),
          nc_(0), mbs_(mini_batch_size), nd_(ndims),
          max_iter_(max_iter), t_(0), lp_(lp), eps_(eps < 0 ? -std::numeric_limits<float>::infinity(): eps),
          avg_size_(avg_size < 0 ? 10: avg_size), project_(project), scale_(scale), bias_(bias)
    {
        sparse_load(path);
    }
    size_t get_nsamples() const {return ns_;}
    size_t get_ndims()    const {return nd_;}
    size_t ndims()        const {return nd_;}
    bool get_bias()       const {return bias_;}
    auto  &get_matrix()         {return m_;}


private:
    void normalize_labels() {
        std::set<int> set;
        for(auto &pair: class_name_map_) set.insert(pair.first);
        std::vector<int> vec(std::begin(set), std::end(set));
        std::sort(std::begin(vec), std::end(vec));
        if(vec.size() != 2)
            throw std::runtime_error(
                std::string("raise NotImplementedError(\"Only binary "
                            "classification currently supported. "
                            "Number of classes found: ") +
                            std::to_string(nc_) + + "\")");
        //std::fprintf(stderr, "Map: {%i: %i, %i: %i}", vec[0], -1, vec[1], 1);
        std::unordered_map<int, int> map;
        map[vec[0]] = -1, map[vec[1]] = 1;
        for(auto &i: v_) {
            i = map[i];
        }
        decltype(class_name_map_) new_cmap;
        for(auto &pair: class_name_map_) new_cmap[map[pair.first]] = pair.second;
        class_name_map_ = std::move(new_cmap);
        nc_ = map.size();
#if !NDEBUG
        for(const auto i: v_) assert(i == -1 || i == 1);
        std::unordered_map<int, int> label_counts;
        for(const auto i: v_) ++label_counts[i];
        for(auto &pair: label_counts) cerr << "Label " << pair.first << " occurs " << pair.second << "times.\n";
#endif
    }
    void load_data(const char *path) {
        dims_t dims(path);
        ns_ = dims.ns_; nd_ = dims.nd_;
        std::tie(m_, v_, class_name_map_) = parse_problem<FloatType, int>(path, dims);
        if(bias_) ++nd_; // bias term
#if !NDEBUG
        if(m_.rows() < 1000) cout << "Input matrix: \n" << m_ << '\n';
#endif
        // Normalize v_
        normalize_labels();
        //init_weights();
        w_ = WMType(nd_, nc_ == 2 ? 1: nc_, lambda_);
        w_.weights_ = 0.;
#if !NDEBUG
        if(v_.size() < 1000) cout << "Input labels: \n" << v_ << '\n';
#endif
        normalize();
        if(nc_ != 2)
            throw std::runtime_error(
                std::string("Number of classes must be 2. Found: ") +
                            std::to_string(nc_));
        LOG_DEBUG("Number of datapoints: %zu. Number of dimensions: %zu\n", ns_, nd_);
    }
    void sparse_load(const char *path) {
        if(bias_) ++nd_; // bias term.
        gzFile fp(gzopen(path, "rb"));
        if(fp == nullptr)
            throw std::runtime_error(std::string("Could not open file at ") + path);
        ks::KString line;
        line.resize(1 << 12);
        size_t linenum(0);
        char *p;

        // Get number of samples.
        ns_ = 0;
        {
            char buf[1 << 18];
            while((p = gzgets(fp, buf, sizeof buf))) {
                switch(*p) case '\n': case '\0': case '#': continue;
                ++ns_;
            }
        }
        gzrewind(fp); // Rewind
#if !NDEBUG
        cerr << "ns: " << ns_ << '\n';
        cerr << "nd: " << nd_ << '\n';
#endif

        m_ = DynamicMatrix<FloatType>(ns_, nd_);
        v_ = DynamicVector<int>(ns_);
        m_ = 0.; // bc sparse, unused entries are zero.
        std::string class_name;
        int  class_id(0);
        int c, moffsets(16), *offsets((int *)malloc(moffsets * sizeof(int)));
        std::unordered_map<std::string, int> tmpmap;
        while((c = gzgetc(fp)) != EOF) {
            if(c != '\n') {
                line.putc_(c);
                continue;
            }
            line->s[line->l] = 0;
            if(line[0] == '#' || line[0] == '\n') {
                line.clear();
                continue;
            }
            const int ntoks(ksplit_core(line.data(), 0, &moffsets, &offsets));
            class_name = line.data() + offsets[0];
            auto m(tmpmap.find(class_name));
            if(m == tmpmap.end()) m = tmpmap.emplace(class_name, class_id++).first;
            v_[linenum] = m->second;
            for(int i(1); i < ntoks; ++i) {
                p = line.data() + offsets[i];
                char *q(strchr(p, ':'));
                if(q == nullptr) throw std::runtime_error("Malformed sparse file.");
                *q++ = '\0';
                const int index(atoi(p) - 1);
                assert(linenum < m_.rows());
                m_(linenum, index) = atof(q);
            }
            ++linenum;
            line.clear();
        }
        free(offsets);
        gzclose(fp);
        for(const auto &pair: tmpmap)
            class_name_map_.emplace(pair.second, pair.first);
        normalize_labels();
        normalize();
        if(nc_ != 2)
            throw std::runtime_error(
                std::string("Number of classes must be 2. Found: ") +
                            std::to_string(nc_));
        w_ = WMType(nd_, nc_ == 2 ? 1: nc_, lambda_);
        w_.weights_ = 0.;
    }
    void normalize() {
        if(scale_) {
            rescale();
        }
        if(bias_) column(m_, nd_ - 1) = 1.; // Bias term
    }
    template<typename RowType>
    void rescale_point(RowType &r) const {
        for(size_t i(0); i < nd_ - static_cast<int>(bias_); ++i) {
            r[i] = (r[i] - r_(i, 0)) * r_(i, 1);
        }
    }
    void rescale() {
        r_ = MatrixKind(nd_ - static_cast<int>(bias_), 2);
        // Could/Should rewrite with pthread-type parallelization and get better memory access pattern.
        #pragma omp parallel for schedule(dynamic)
        for(size_t i = 0; i < nd_ - static_cast<int>(bias_); ++i) {
            auto col(column(m_, i));
            FloatType stdev_inv, colmean;
            assert(ns_ == col.size());
            FloatType sum(0.);
            for(auto c: col) sum += c;
            r_(i, 0) = colmean = sum / ns_;
            r_(i, 1) = stdev_inv = 1. / std::sqrt(variance(col, colmean));
            for(auto cit(col.begin()), cend(col.end()); cit != cend; ++cit)
                *cit = (*cit - colmean) * stdev_inv;
            cerr << "Column " << i + 1 << " has mean " << colmean << " and stdev " << 1./stdev_inv << '\n';
            cerr << "New variance: " << variance(col) << ". New mean: " << mean(col) <<'\n';
        }
    }
    template<typename RowType>
    double predict(const RowType &datapoint) const {
        return dot(row(w_.weights_, 0), datapoint);
    }
public:
    template<typename RowType>
    int classify_external(RowType &data) const {
        if(r_.rows()) rescale_point(data);
        return classify(data);
    }
    template<typename RowType>
    int classify(const RowType &data) const {
        static const int tbl[]{-1, 1};
        const double pred(predict(data));
        return tbl[pred > 0.];
    }
    FloatType loss() const {
        size_t mistakes(0);
        for(size_t index(0); index < ns_; ++index)
            mistakes += (classify(row(m_, index)) != v_[index]);
        return static_cast<double>(mistakes) / ns_;
    }
    void train() {
#if !NDEBUG
        const size_t interval(max_iter_ / 10);
#endif
        size_t avgs_used(0);
        decltype(w_.weights_) tmpsum(1, nd_);
        decltype(w_.weights_) last_weights(1, nd_);
        auto wrow(row(w_.weights_, 0));
        auto trow(row(tmpsum, 0));
        const size_t max_end_index(std::min(ns_ - mbs_, ns_));
        std::vector<size_t> indices;
        for(t_ = 0; avgs_used < avg_size_; ++t_) {
#if !NDEBUG
            if((t_ % interval) == 0) {
                const double ls(loss()), current_norm(w_.get_norm_sq());
                cerr << "Loss: " << ls * 100 << "%" << " at time = " << t_ << 
                        " with norm of w = " << current_norm << ".\n";
            }
#endif
            const double eta(lp_(t_));
            tmpsum = 0.; // reset to 0 each time.
            indices.clear();
            #pragma omp parallel for
            for(size_t i = 0; i < mbs_; ++i) {
                const size_t index(fastrangesize(rand64(), max_end_index));
                if(predict(row(m_, index)) * v_[index] < 1.) {
                    #pragma omp critical
                    row(tmpsum, 0) += row(m_, index) * v_[index];
                }
                // Could be made more cache-friendly by randomly selecting and
                // processing chunks of the data, but for now this is fine.
            }
            const size_t nels_added(mbs_);
            if(t_ < max_iter_) last_weights = w_.weights_;
            w_.scale(1.0 - eta * lambda_);
            wrow += trow * (eta / nels_added);
            if(project_) {
                const double norm(w_.get_norm_sq());
                if(norm > 1. / lambda_)
                    w_.scale(std::sqrt(1.0 / (lambda_ * norm)));
            }
            if(t_ >= max_iter_ || diffnorm(row(last_weights, 0), row(w_.weights_, 0)) < eps_) {
                if(w_avg_.weights_.rows() == 0) w_avg_ = WMType(nd_, nc_ == 2 ? 1: nc_, lambda_);
                w_avg_.weights_ = 0.;
                row(w_avg_.weights_, 0) += wrow;
                ++avgs_used;
            }
        }
        row(w_avg_.weights_, 0) *= 1. / avg_size_;
        double ls(loss());
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
}; // LinearSVM

} // namespace svm
#if !NDEBUG
#  undef NOTIFICATION_INTERVAL
#endif


#endif // _LINEAR_SVM_H_
