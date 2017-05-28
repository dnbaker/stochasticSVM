#include "src/train_svm.h"

#define LAPLACE_SWITCH case 'S': sigma = atof(optarg); break;
#define KERNEL_PARAMS FLOAT_TYPE sigma(1.0);

#define KERNEL_INIT LaplacianKernel<FLOAT_TYPE> kernel(sigma)
DECLARE_KERNEL_SVM(KERNEL_INIT,\
                   LAPLACE_SWITCH,\
                   KERNEL_PARAMS,\
                   "-S\tSigma for Laplace kernel [1.0] \n", "S:")
