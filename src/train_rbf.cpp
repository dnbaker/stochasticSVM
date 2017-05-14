#include "src/train_kernel.h"



#define RBF_SWITCH case 'g': gamma = atof(optarg); break;
#define KERNEL_PARAMS FLOAT_TYPE gamma(1.0);

#define KERNEL_INIT RBFKernel<FLOAT_TYPE> kernel(gamma)
DECLARE_KERNEL_SVM(KERNEL_INIT,\
                   RBF_SWITCH,\
                   KERNEL_PARAMS,\
                   "-g\tGamma for rbf kernel [1.0] \n", "g:")
