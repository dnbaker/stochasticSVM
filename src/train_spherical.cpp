#include "src/train_svm.h"

#define SPHERICAL_SWITCH case 'S': sigma = atof(optarg); break;
#define KERNEL_PARAMS FLOAT_TYPE sigma(1.0);

#define KERNEL_INIT SphericalKernel<FLOAT_TYPE> kernel(sigma)
DECLARE_KERNEL_SVM(KERNEL_INIT,\
                   SPHERICAL_SWITCH,\
                   KERNEL_PARAMS,\
                   "-S\tSigma for Spherical kernel [1.0] \n", "S:")
