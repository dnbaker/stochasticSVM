#include "src/train_kernel.h"

#define CIRCULAR_SWITCH case 'S': sigma = atof(optarg); break;
#define KERNEL_PARAMS FLOAT_TYPE sigma(1.0);

#define KERNEL_INIT CircularKernel<FLOAT_TYPE> kernel(sigma)
DECLARE_KERNEL_SVM(KERNEL_INIT,\
                   CIRCULAR_SWITCH,\
                   KERNEL_PARAMS,\
                   "-S\tSigma for Circular kernel [1.0] \n", "S:")
