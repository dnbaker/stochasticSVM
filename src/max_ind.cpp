#include "train_svm.h"

using std::cerr;

int main(int argc, char *argv[]) {
    if(argc == 1) throw std::runtime_error(std::string("Usage: ") + argv[0] + " <file>");
    int max(0);
    for(char **q(argv); *q; ++q) max = std::max(max, get_max_ind(*q));
    cerr << "Max index: " << max << '\n';
}
