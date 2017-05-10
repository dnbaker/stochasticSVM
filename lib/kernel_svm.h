#ifndef _KERNEL_SVM_H_
#define _KERNEL_SVM_H_
#include "lib/linear_svm.h"
#include "lib/kernel.h"
#include "fastrange/fastrange.h"

namespace svm {

template<class Kernel,
         typename MatrixType=float,
         class MatrixKind=DynamicMatrix<MatrixType>,
         typename VectorType=int,
         class LearningPolicy=PegasosLearningRate<MatrixType>>
class KernelSVM {

    // Increase nd by 1 and set all the last entries to "1" to add
    // the bias term implicitly.
    // (See http://ttic.uchicago.edu/~nati/Publications/PegasosMPB.pdf,
    //  section 6.)

    using WMType = WeightMatrix<MatrixType>;

    MatrixKind                m_; // Training Data
    // Weights. one-dimensional for 2-class, nc_-dimensional for more.
    WMType w_;
    WMType w_avg_;
    DynamicVector<VectorType> a_;
    DynamicVector<VectorType> v_; // Labels
#if RENORMALIZE
    MatrixKind                r_; // Renormalization values. Subtraction, then multiplication
                                  // Not required: not used.
#endif
    const MatrixType     lambda_; // Lambda Parameter
    const Kernel         kernel_;
    size_t                   nc_; // Number of classes
    const size_t            mbs_; // Mini-batch size
    size_t                   ns_; // Number samples
    size_t                   nd_; // Number of dimensions
    const size_t       max_iter_; // Maximum iterations.
    size_t                    t_; // Timepoint.
    const LearningPolicy     lp_; // Calculates learning rate at a timestep t.
    const MatrixType        eps_; // epsilon termination.
    const size_t       avg_size_; // Number to average at end.
    const bool          project_; // Whether or not to perform projection step.
    std::unordered_map<VectorType, std::string> class_name_map_;

public:
    // Dense constructor
    KernelSVM(const char *path,
               const MatrixType lambda,
               LearningPolicy lp,
               Kernel kernel=LinearKernel<MatrixType>(),
               size_t mini_batch_size=256uL,
               size_t max_iter=100000,  const MatrixType eps=1e-6,
               long avg_size=-1, bool project=false)
        : lambda_(lambda), kernel_(std::move(kernel)),
          nc_(0), mbs_(mini_batch_size),
          max_iter_(max_iter), t_(0), lp_(lp), eps_(eps),
          avg_size_(avg_size < 0 ? 1000: avg_size),
          project_(project)
    {
        load_data(path);
    }
    KernelSVM(const char *path, size_t ndims,
               const MatrixType lambda,
               LearningPolicy lp,
               Kernel kernel=LinearKernel<MatrixType>(),
               size_t mini_batch_size=256uL,
               size_t max_iter=100000,  const MatrixType eps=1e-6,
               long avg_size=-1, bool project=false)
        : lambda_(lambda), kernel_(std::move(kernel)),
          nc_(0), mbs_(mini_batch_size), nd_(ndims),
          max_iter_(max_iter), t_(0), lp_(lp), eps_(eps),
          avg_size_(avg_size < 0 ? 10: avg_size), project_(project)
    {
        sparse_load(path);
    }
    size_t get_nsamples() {return ns_;}
    size_t get_ndims()    {return nd_;}
    auto  &get_matrix()   {return m_;}


private:
    void normalize_labels() {
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
#if !NDEBUG
        for(const auto i: v_) assert(i == -1 || i == 1);
#endif
    }
    void load_data(const char *path) {
        dims_t dims(path);
        ns_ = dims.ns_; nd_ = dims.nd_;
        std::tie(m_, v_, class_name_map_) = parse_problem<MatrixType, VectorType>(path, dims);
        ++nd_;
        if(m_.rows() < 1000) cout << "Input matrix: \n" << m_ << '\n';
        // Normalize v_
        normalize_labels();
        //init_weights();
        w_ = WMType(nd_, nc_ == 2 ? 1: nc_, lambda_);
        w_.weights_ = 0.;
        if(v_.size() < 1000) cout << "Input labels: \n" << v_ << '\n';
        normalize();
        LOG_DEBUG("Number of datapoints: %zu. Number of dimensions: %zu\n", ns_, nd_);
    }
    void sparse_load(const char *path) {
        ++nd_; // bias term, in case used.
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

        m_ = DynamicMatrix<MatrixType>(ns_, nd_);
        v_ = DynamicVector<VectorType>(ns_);
        m_ = 0.; // bc sparse, unused entries are zero.
        std::string class_name;
        VectorType  class_id(0);
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
            if(m == tmpmap.end()) {
                tmpmap.emplace(class_name, class_id++);
            }
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
        column(m_, nd_ - 1) = 1.; // Bias term
    }
    template<typename RowType>
    double predict(const RowType &datapoint) const {
        return dot(row(w_.weights_, 0), datapoint);
    }
    template<typename WeightMatrixKind>
    void add_entry(const size_t index, WeightMatrixKind &tmpsum, size_t &nels_added) {
        if(predict(row(m_, index)) * v_[index] < 1.) row(tmpsum, 0) += row(m_, index) * v_[index];
        ++nels_added;
    }
    template<typename RowType>
    VectorType classify(const RowType &data) const {
        static const VectorType tbl[]{-1, 1};
        const double pred(predict(data));
        return tbl[pred > 0.];
    }
public:
    MatrixType loss() const {
        size_t mistakes(0);
        for(size_t index(0); index < ns_; ++index) {
            const VectorType c(classify(row(m_, index)));
            if(c != v_[index]) {
                ++mistakes;
            }
        }
        return static_cast<double>(mistakes) / ns_;
    }
    void train() {
        size_t avgs_used(0);
        decltype(w_.weights_) tmpsum(1, nd_);
        decltype(w_.weights_) last_weights(1, nd_);
        size_t nels_added(0);
        auto wrow(row(w_.weights_, 0));
        auto trow(row(tmpsum, 0));
        const size_t max_end_index(std::min(ns_ - mbs_, ns_));
        for(t_ = 0; avgs_used < avg_size_; ++t_) {
            nels_added = 0;
#if !NDEBUG
            if((t_ % NOTIFICATION_INTERVAL) == 0) {
                const double ls(loss()), current_norm(w_.get_norm_sq());
                cerr << "Loss: " << ls * 100 << "%" << " at time = " << t_ << 
                        " with norm of w = " << current_norm << ".\n";
            }
#endif
            const double eta(lp_(t_));
            tmpsum = 0.; // reset to 0 each time.
            for(size_t i(0); i < mbs_; ++i) {
                const size_t start_index = fastrangesize(rand64(), max_end_index);
                add_entry(start_index, tmpsum, nels_added);
            }
            if(t_ < max_iter_) {
                last_weights = w_.weights_;
            }
            w_.scale(1.0 - eta * lambda_);
            wrow += trow * (eta / nels_added);
            if(project_) {
                const double norm(w_.get_norm_sq());
                if(norm > 1. / lambda_)
                    w_.scale(std::sqrt(1.0 / (lambda_ * norm)));
            }
            if(t_ >= max_iter_ || diffnorm(row(last_weights, 0), row(w_.weights_, 0)) < eps_) { // TODO: replace false with epsilon
                if(w_avg_.weights_.rows() == 0) w_avg_ = WMType(nd_, nc_ == 2 ? 1: nc_, lambda_);
                w_avg_.weights_ = 0.;
                row(w_avg_.weights_, 0) += wrow;
                ++avgs_used;
            }
        }
        row(w_avg_.weights_, 0) *= 1. / avg_size_;
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
        fprintf(fp, "#%s\n", kernel_.str().data());
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


#endif // _KERNEL_SVM_H_
