#ifndef _PROBLEM_H_
#define _PROBLEM_H_
#include "lib/misc.h"
#include "blaze/Math.h"
#include "lib/parse.h"
#include "lib/kernel.h"
#include "lib/mkernel.h"
#include "lib/mathutil.h"
#include "lib/learning_rate.h"
#include "fastrange/fastrange.h"

namespace svm {

#define NOTIFICATION_INTERVAL (1uLL)

// TODONE: Polynomial kernels
// TODONE: Gradients

template<typename MatrixType, typename WeightMatrixKind=DynamicMatrix<MatrixType>>
class WeightMatrix {
    double norm_;
public:
    WeightMatrixKind weights_;
    operator WeightMatrixKind&() {return weights_;}
    operator const WeightMatrixKind&() const {return weights_;}
    WeightMatrix(size_t ns, size_t nc, MatrixType lambda):
        norm_{0.}, weights_{WeightMatrixKind(nc == 2 ? 1: nc, ns)} {
        cerr << "Initialized WeightMatrix with " << nc << " classes and " << ns << " samples\n";
    }
    WeightMatrix(): norm_(0.) {}
    void scale(MatrixType factor) {
        norm_ *= factor * factor;
        weights_ *= factor;
    }
    MatrixType get_norm_sq() {
        return norm_ = dot(row(weights_, 0), row(weights_, 0));
    }
    double norm() const {return norm_;}
};

template<class Kernel,
         typename MatrixType=float,
         class MatrixKind=DynamicMatrix<MatrixType>,
         typename VectorType=int,
         class LearningPolicy=PegasosLearningRate<MatrixType>>
class SVMTrainer {

    // Increase nd by 1 and set all the last entries to "1" to add
    // the bias term implicitly.
    // (See http://ttic.uchicago.edu/~nati/Publications/PegasosMPB.pdf,
    //  section 6.)

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
    // Dense constructor
    SVMTrainer(const char *path,
               const MatrixType lambda,
               LearningPolicy lp,
               Kernel kernel=LinearKernel<MatrixType>(),
               size_t mini_batch_size=1,
               size_t max_iter=100000,  const MatrixType eps=1e-12,
               long avg_size=-1)
        : lambda_(lambda), kernel_(std::move(kernel)),
          nc_(0), mbs_(mini_batch_size),
          max_iter_(max_iter), t_(0), lp_(lp), eps_(eps),
          avg_size_(avg_size < 0 ? 1000: avg_size)
    {
        cerr << "Dense loader!\n";
        load_data(path);
    }
    SVMTrainer(const char *path, size_t ndims,
               const MatrixType lambda,
               LearningPolicy lp,
               Kernel kernel=LinearKernel<MatrixType>(),
               size_t mini_batch_size=1,
               size_t max_iter=100000,  const MatrixType eps=1e-12,
               long avg_size=-1)
        : lambda_(lambda), kernel_(std::move(kernel)),
          nc_(0), mbs_(mini_batch_size), nd_(ndims),
          max_iter_(max_iter), t_(0), lp_(lp), eps_(eps),
          avg_size_(avg_size < 0 ? 1000: avg_size)
    {
        cerr << "Sparse loader with " << ndims << " dimensions\n";
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
        for(auto &pair: class_name_map_) {
            cerr << "Key: " << pair.second << " value: " << pair.first << '\n';
        }
        for(const auto i: v_) assert(i == -1 || i == 1);
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
        cerr << "ns: " << ns_ << '\n';
        cerr << "nd: " << nd_ << '\n';
        gzrewind(fp); // Rewind

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
        for(const auto &pair: tmpmap) class_name_map_.emplace(pair.second, pair.first);
        cerr << "tmpmap size: " << tmpmap.size() << '\n';
        free(offsets);
        gzclose(fp);
        normalize_labels();
        normalize();
        w_ = WMType(nd_, nc_ == 2 ? 1: nc_, lambda_);
        w_.weights_ = 0.;
        cerr << "parsed in! Number of rows in weights? " << w_.weights_.rows() << '\n';
        LOG_INFO("Norm of weights beginning? %lf\n", w_.get_norm_sq());
    }
    void normalize() {
#if RENORMALIZE
        // Unneeded according to paper? (??? maybe only unneeded for sparse matrices?)
        // Not set
        r_ = MatrixKind(nd_, 2);
        // Could/Should rewrite with pthread-type parallelization and get better memory access pattern.
        #pragma omp parallel for schedule(dynamic)
        for(size_t i = 0; i < nd_ - 1; ++i) {
            auto col(column(m_, i));
            assert(ns_ == col.size());
            double sum(0.);
            for(auto c: col) sum += c;
            MatrixType mean(sum / ns_);
            r_(i, 0) = mean;
            const auto var(variance(col, mean));
            r_(i, 1) = 1. / (MatrixType)std::sqrt(var);
            for(auto cit(col.begin()), cend(col.end()); cit != cend; ++cit)
                *cit = (*cit - mean) * r_(i, 1);
        }
        double absum(0.);
        for(size_t j(0); j < ns_; ++j)
           for(size_t i(0); i < nd_ - 1; ++i)
                absum += abs(m_(j, i));
        absum /= (ns_ * nd_ - 1);
        column(m_, nd_ - 1) = absum; // Bias term -- use mean absolute value in matrix.
#elif SETTING_TO_ONE
        column(m_, nd_ - 1) = 1.;
#else
        column(m_, nd_ - 1) = 0.;
#endif
    }
    template<typename RowType>
    double predict_linear(const RowType &datapoint) const {
        const auto ret(dot(row(w_.weights_, 0), datapoint));
        return ret;
    }
    template<typename WeightMatrixKind>
    void add_entry_linear(const size_t index, WeightMatrixKind &tmpsum, size_t &nels_added) {
        auto mrow(row(m_, index));
        MatrixType pred;
        if((pred = predict_linear(mrow) * v_[index]) < 1.) {
            row(tmpsum, 0) += mrow * v_[index];
        }
        ++nels_added;
    }
    template<typename RowType>
    VectorType classify(const RowType &data) const {
        static const VectorType tbl[]{-1, 1};
        const double pred(predict_linear(data));
        if(nc_ == 2) {
            return tbl[pred < 0];
        } else {
            throw std::runtime_error(std::string("NotImplementedError: number of classes: ")  + std::to_string(nc_));
        }
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
    void train_linear() {
        cerr << "Starting to train\n";
        cerr << "Matrix: \n" << m_;
        cerr << "Labels: \n" << v_;
        exit(1);
        size_t avgs_used(0);
        decltype(w_.weights_) tmpsum(1, nd_);
        size_t nels_added(0);
        auto wrow(row(w_.weights_, 0));
        auto trow(row(tmpsum, 0));
        cerr << ("Set wrow and trow\n");
        const size_t max_end_index(std::min(ns_ - mbs_, ns_));
        for(t_ = 0; avgs_used < avg_size_; ++t_) {
            nels_added = 0;
            if((t_ % NOTIFICATION_INTERVAL) == 0) {
                const double ls(loss()), current_norm(w_.get_norm_sq());
                cerr << "Loss: " << ls * 100 << "%" << " at time = " << t_ << 
                        " with norm of w = " << current_norm << ".\n";
            }
            const double eta(lp_(t_));
            tmpsum = 0.; // reset to 0 each time.
            for(size_t i(0); i < mbs_; ++i) {
                const size_t start_index = fastrangesize(rand64(), max_end_index);
                add_entry_linear(start_index, tmpsum, nels_added);
            }
            w_.scale(1.0 - eta * lambda_);
            wrow += trow * (eta / nels_added);
            const double norm(w_.get_norm_sq());
            if(norm > 1. / lambda_) {
                LOG_DEBUG("Scaling down bc too big\n");
                w_.scale(std::sqrt(1.0 / (lambda_ * norm)));
            }
            if(t_ >= max_iter_ || false) { // TODO: replace false with epsilon
                if(w_avg_.weights_.rows() == 0) w_avg_ = WMType(nd_, nc_ == 2 ? 1: nc_, lambda_);
                auto avg_row(row(w_avg_.weights_, 0));
                LOG_DEBUG("Updating averages. t_: %zu. max: %zu\n", t_, max_iter_);
                avg_row += wrow;
                ++avgs_used;
            }
            //LOG_DEBUG("Finishing loop\n");
        }
        row(w_avg_.weights_, 0) *= 1. / avg_size_;
        LOG_DEBUG("Trained!\n");
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
