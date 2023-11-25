#include <torch/extension.h>

#include <torch/library.h>
#include <c10/core/DispatchKey.h>

#include "cudnn_convolution.h"

namespace sfast {
namespace operators {

using namespace torch;

void initCUDNNConvolutionBindings(torch::Library &m) {
#if defined(WITH_CUDA)
  m.def("cudnn_convolution_bias_add",
        dispatch(c10::DispatchKey::CompositeImplicitAutograd,
                 cudnn_convolution_bias_add));
  m.def("cudnn_convolution_bias",
        dispatch(c10::DispatchKey::CompositeImplicitAutograd,
                 cudnn_convolution_bias));
  m.def("cudnn_convolution_bias_sigmoid",
        dispatch(c10::DispatchKey::CompositeImplicitAutograd,
                 cudnn_convolution_bias_sigmoid));
  m.def("cudnn_convolution_bias_relu",
        dispatch(c10::DispatchKey::CompositeImplicitAutograd,
                 cudnn_convolution_bias_relu));
  m.def("cudnn_convolution_bias_tanh",
        dispatch(c10::DispatchKey::CompositeImplicitAutograd,
                 cudnn_convolution_bias_tanh));
  m.def("cudnn_convolution_bias_add_sigmoid",
        dispatch(c10::DispatchKey::CompositeImplicitAutograd,
                 cudnn_convolution_bias_add_sigmoid));
  m.def("cudnn_convolution_bias_add_relu",
        dispatch(c10::DispatchKey::CompositeImplicitAutograd,
                 cudnn_convolution_bias_add_relu));
  m.def("cudnn_convolution_bias_add_tanh",
        dispatch(c10::DispatchKey::CompositeImplicitAutograd,
                 cudnn_convolution_bias_add_tanh));
#endif
}

} // namespace operators
} // namespace sfast
