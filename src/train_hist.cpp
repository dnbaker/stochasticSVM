#include "src/train_svm.h"

#define RATIONAL_SWITCH
#define KERNEL_PARAMS

#define KERNEL_INIT HistogramKernel<FLOAT_TYPE> kernel;
DECLARE_KERNEL_SVM(KERNEL_INIT,\
                   RATIONAL_SWITCH,\
                   KERNEL_PARAMS,\
                   "", "")
