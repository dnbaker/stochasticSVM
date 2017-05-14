#include "src/train_kernel.h"

#define TANH_INIT(kernel, kappa, tanhc) TanhKernel<FLOAT_TYPE> kernel(kappa, tanhc)
#define TANH_PARAMS(kappa, tanhc) FLOAT_TYPE kappa(1.0); FLOAT_TYPE tanhc(0.)
#define TANH_SWITCH case 'k': kappa = atof(optarg); break; case 'c': tanhc = atof(optarg); break;
#define KERNEL_PARAMS FLOAT_TYPE kappa(1.0), tanhc(0.0);

#define KERNEL_INIT TanhKernel<FLOAT_TYPE> kernel(kappa, tanhc)
DECLARE_KERNEL_SVM(KERNEL_INIT,\
                   TANH_SWITCH,\
                   KERNEL_PARAMS,\
                   "-k\tKappa for tanh kernel\n-c\tC for tanh kernel\n", "k:c:")
