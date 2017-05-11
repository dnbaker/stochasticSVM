#ifndef _KERNEL_SVM_H_
#define _KERNEL_SVM_H_
#include "lib/linear_svm.h"
#include "lib/kernel.h"
#include "lib/khash64.h"
#include "fastrange/fastrange.h"
#include <unordered_set>
#include <mutex>

namespace svm {

KHASH_SET_INIT_INT64(I) // 64-bit set for randomly selected batch sizes.

template<class Kernel,
         typename FloatType=float,
         class MatrixKind=DynamicMatrix<FloatType>>
class KernelSVM {

    // Increase nd by 1 and set all the last entries to "1" to add
    // the bias term implicitly.
    // (See http://ttic.uchicago.edu/~nati/Publications/PegasosMPB.pdf,
    //  section 6.)

    MatrixKind                   m_; // Training Data
    // Weights. one-dimensional for 2-class, nc_-dimensional for more.
    CompressedVector<int>        a_;
    DynamicVector<int>           v_; // Labels
#if RENORMALIZE
    MatrixKind                   r_; // Renormalization values. Subtraction, then multiplication
                                  // Not required: not used.
#endif
    const FloatType        lambda_; // Lambda Parameter
    const Kernel            kernel_;
    //DynamicMatrix<FloatType>     w_; // Final weights: only used at completion.
    size_t                      nc_; // Number of classes
    const size_t               mbs_; // Mini-batch size
    size_t                      ns_; // Number samples
    size_t                      nd_; // Number of dimensions
    const size_t          max_iter_; // Maximum iterations.
    size_t                       t_; // Timepoint.
    const FloatType           eps_; // epsilon termination.
    std::unordered_map<int, std::string> class_name_map_;
    khash_t(I)                  *h_; // Hash set of elements being used.

public:
    // Dense constructor
    KernelSVM(const char *path,
              const FloatType lambda,
              Kernel kernel=LinearKernel<FloatType>(),
              size_t mini_batch_size=256uL,
              size_t max_iter=100000,  const FloatType eps=1e-6)
        : lambda_(lambda), kernel_(std::move(kernel)),
          nc_(0), mbs_(mini_batch_size),
          max_iter_(max_iter), t_(0), eps_(eps), h_(kh_init(I))
    {
        load_data(path);
    }
    KernelSVM(const char *path, size_t ndims,
              const FloatType lambda,
              Kernel kernel=LinearKernel<FloatType>(),
              size_t mini_batch_size=256uL,
              size_t max_iter=100000,  const FloatType eps=1e-6)
        : lambda_(lambda), kernel_(std::move(kernel)),
          nc_(0), mbs_(mini_batch_size), nd_(ndims),
          max_iter_(max_iter), t_(0), eps_(eps), h_(kh_init(I))
    {
        sparse_load(path);
    }
    ~KernelSVM() {kh_destroy(I, h_);}
    size_t get_nsamples() {return ns_;}
    size_t get_ndims()    {return nd_;}
    auto  &get_matrix()   {return m_;}


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
        std::unordered_map<int, int> map;
        //std::fprintf(stderr, "Map: {%i: %i, %i: %i}", vec[0], -1, vec[1], 1);
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
        ++nd_;
        if(m_.rows() < 1000) cout << "Input matrix: \n" << m_ << '\n';
        // Normalize v_
        normalize_labels();
        //init_weights();
        if(v_.size() < 1000) cout << "Input labels: \n" << v_ << '\n';
        normalize();
        if(nc_ != 2)
            throw std::runtime_error(
                std::string("Number of classes must be 2. Found: ") +
                            std::to_string(nc_));
        a_ = CompressedVector<int>(ns_, ns_ >> 1);
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
        a_ = CompressedVector<int>(ns_, ns_ >> 1);
    }
    void normalize() {
        column(m_, nd_ - 1) = 1.; // Bias term
    }
    template<typename RowType>
    double predict(const RowType &datapoint) const {
        double ret(0.);
        if(t_ == 0) cerr << "nonZeros: " << a_.nonZeros() << '\n';
        for(auto it(a_.cbegin()), e(a_.cend()); it != e; ++it) {
            if(kh_get(I, h_, it->value()) == kh_end(h_)) {
                ret += v_[it->index()] * it->value() * kernel_(row(m_, it->index()), datapoint);
            }
        }
        if(t_ == 0) cerr << "ret: " << ret << '\n';
        return ret;
    }
    double predict(size_t index) const {
        return predict(row(m_, index));
    }
    template<typename RowType>
    int classify(const RowType &data) const {
        static const int tbl[]{-1, 1};
        return tbl[predict(data) > 0.];
    }
public:
    FloatType loss() const {
        size_t mistakes(0);
        for(size_t index(0); index < ns_; ++index) {
            mistakes += (classify(index) != v_[index]);
        }
        return static_cast<double>(mistakes) / ns_;
    }
    void train() {
        decltype(a_) last_alphas;
        kh_resize(I, h_, mbs_ * 1.5);
        kh_clear(I, h_);
        std::set<size_t> indices;
        for(t_ = 0; t_ < max_iter_; ++t_) {
            indices.clear();
            last_alphas = a_;
            int khr;
            size_t ndiff(0);
            //std::vector<std::mutex> muts(nd_ >> 6);
            for(size_t i = 0; i < mbs_; ++i) {
                // Could probably speed up by changing loop iteration:
                // Iterating through all alphas and processing each element for it.
                kh_put(I, h_, fastrangesize(rand64(), ns_), &khr);
            }
            if(t_ == 0) assert(a_.nonZeros() == 0);
            #pragma omp parallel for
            for(khiter_t ki = 0; ki < kh_end(h_); ++ki) {
                if(kh_exist(h_, ki)) {
                    const double prediction(predict(kh_key(h_, ki)));
                    if(prediction * v_[kh_key(h_, ki)] < 1.) {
                        #pragma omp critical
                        indices.insert(kh_key(h_, ki));
                        ++ndiff;
                    }
                }
            }
            if(t_ == 0) assert(a_.nonZeros() == 0);
            for(const auto index: indices) ++a_[index];
            kh_clear(I, h_);
            cerr << "loss: " << loss() * 100 << "%\n"
                 << "nonzeros: " << nonZeros(a_)
                 << " Number incremented (bc < 1): " << ndiff
                 << " iteration: " << t_ << '\n';
            const double dn(diffnorm(a_, last_alphas));
            //cerr << "Diff norm: " << dn << '\n';
            if(dn < eps_) break;
            // If the results are the same (or close enough).
            // This should probably be updated to reflect the weight components
            // involved in the norm of the difference. Minor detail, however.
        }
        cleanup();
    }
    void cleanup() {
        cerr << "final loss: " << loss() * 100 << "%. Number of iterations: " << t_ << "\n";
#if 0
        /*
        This can only work if we apply a Taylor expansion to our support vectors
        and our input data, which makes it such that we can't free memory like
        we'd like to.
        w_ = DynamicMatrix<FloatType>(1, nd_);
        auto wrow = row(w_, 0);
        wrow = 0.;
        cerr << "Size of vector: " << a_.size() << '\n';
        cerr << "Rows in matrix: " << m_.rows()  << '\n';
        assert(a_.size() == m_.rows());
        for(auto it(a_.cbegin()), end(a_.cend()); it != end; ++it)
            wrow += a_[it->index()] * v_[it->value()] *
                    row(m_, it->index());
        w_ *= 1. / (lambda_ * (t_ - 1));
        free_matrix(m_);
        free_vector(v_);
        */
#endif
    }
    void write(FILE *fp, bool scientific_notation=false) {
        fprintf(fp, "#Dimensions: %zu.\n", nd_);
        fprintf(fp, "#%s\n", kernel_.str().data());
        ks::KString line;
        line.resize(5 * a_.nonZeros());
        const char *fmt(scientific_notation ? "%e, ": "%f, ");
        for(const auto &it: a_) {
            line.sprintf("i:%i,v:%i,y:%i|", it.index(), it.value(), v_[it.index()]);
        }
        line[line.size() - 1] = '\n';
        fwrite(line.data(), line.size(), 1, fp);
    }
}; // LinearSVM

} // namespace svm


#endif // _KERNEL_SVM_H_
