#include "src/train_svm.h"

#define RATIONAL_SWITCH case 'c': kc = atof(optarg); break;
#define KERNEL_PARAMS FLOAT_TYPE kc(0.0);

#define KERNEL_INIT RationalQuadKernel<FLOAT_TYPE> kernel(kc)
DECLARE_KERNEL_SVM(KERNEL_INIT,\
                   RATIONAL_SWITCH,\
                   KERNEL_PARAMS,\
                   "-c\t c parameter for rational quadratic kernel [0.0]\n", "c:")
