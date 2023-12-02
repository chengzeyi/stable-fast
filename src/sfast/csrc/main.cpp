#include <torch/extension.h>

#include "jit/init.h"
#include "misc.h"
#include "operators/cudnn/cudnn_convolution.h"
#include "operators/cublas/cublas_gemm.h"
#include "operators/cutlass/cutlass_qlinear.h"
#include "operators/cutlass/cutlass_dual_linear.h"
#include "operators/fused_linear.h"

using namespace sfast;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  jit::initJITBindings(m);
  misc::initMiscBindings(m);
}

TORCH_LIBRARY(sfast, m) {
  operators::initCUDNNConvolutionBindings(m);
  operators::initCUBLASGEMMBindings(m);
  operators::initCutlassQLinearBindings(m);
  operators::initCutlassDualLinearBindings(m);
  operators::initFusedLinearBindings(m);
}
