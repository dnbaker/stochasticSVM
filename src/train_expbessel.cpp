#include "src/train_svm.h"

#define EXP_BESSEL_SWITCH case 'S': sigma = atof(optarg); break; case 'O': order = atoi(optarg); break;
#define KERNEL_PARAMS FLOAT_TYPE order(1), sigma(1.0);

#define KERNEL_INIT ExponentialBesselKernel<FLOAT_TYPE> kernel(sigma, order)
DECLARE_KERNEL_SVM(KERNEL_INIT,\
                   EXP_BESSEL_SWITCH,\
                   KERNEL_PARAMS,\
                   "-S\tSigma for Exponential Bessel Kernel [1.0]\n-O\tOrder for Exponential Bessel Kernel [1]", "S:O:")
