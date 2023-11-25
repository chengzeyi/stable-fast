#pragma once
#include <torch/extension.h>

#include <torch/library.h>

namespace sfast {
namespace operators {

using namespace torch;

#if defined(WITH_CUDA)
Tensor cudnn_convolution_bias_add(const Tensor &input_t, const Tensor &weight_t,
                                  const c10::optional<Tensor> &bias_t,
                                  const c10::optional<Tensor> &z_t,
                                  const c10::optional<Scalar> &alpha,
                                  IntArrayRef stride, IntArrayRef padding,
                                  IntArrayRef dilation, bool transposed,
                                  IntArrayRef output_padding, int64_t groups);

Tensor cudnn_convolution_bias(const Tensor &input_t, const Tensor &weight_t,
                              const c10::optional<Tensor> &bias_t,
                              IntArrayRef stride, IntArrayRef padding,
                              IntArrayRef dilation, bool transposed,
                              IntArrayRef output_padding, int64_t groups);

Tensor cudnn_convolution_bias_sigmoid(const Tensor &input_t,
                                      const Tensor &weight_t,
                                      const c10::optional<Tensor> &bias_t,
                                      IntArrayRef stride, IntArrayRef padding,
                                      IntArrayRef dilation, bool transposed,
                                      IntArrayRef output_padding,
                                      int64_t groups);

Tensor cudnn_convolution_bias_relu(const Tensor &input_t,
                                   const Tensor &weight_t,
                                   const c10::optional<Tensor> &bias_t,
                                   IntArrayRef stride, IntArrayRef padding,
                                   IntArrayRef dilation, bool transposed,
                                   IntArrayRef output_padding, int64_t groups);

Tensor cudnn_convolution_bias_tanh(const Tensor &input_t,
                                   const Tensor &weight_t,
                                   const c10::optional<Tensor> &bias_t,
                                   IntArrayRef stride, IntArrayRef padding,
                                   IntArrayRef dilation, bool transposed,
                                   IntArrayRef output_padding, int64_t groups);

Tensor cudnn_convolution_bias_add_sigmoid(const Tensor &input_t,
                                          const Tensor &weight_t,
                                          const c10::optional<Tensor> &bias_t,
                                          const c10::optional<Tensor> &z_t,
                                          const c10::optional<Scalar> &alpha,
                                          IntArrayRef stride,
                                          IntArrayRef padding,
                                          IntArrayRef dilation,
                                          bool transposed,
                                          IntArrayRef output_padding,
                                          int64_t groups);

Tensor cudnn_convolution_bias_add_relu(const Tensor &input_t,
                                       const Tensor &weight_t,
                                       const c10::optional<Tensor> &bias_t,
                                       const c10::optional<Tensor> &z_t,
                                       const c10::optional<Scalar> &alpha,
                                       IntArrayRef stride, IntArrayRef padding,
                                       IntArrayRef dilation, bool transposed,
                                       IntArrayRef output_padding,
                                       int64_t groups);

Tensor cudnn_convolution_bias_add_tanh(const Tensor &input_t,
                                       const Tensor &weight_t,
                                       const c10::optional<Tensor> &bias_t,
                                       const c10::optional<Tensor> &z_t,
                                       const c10::optional<Scalar> &alpha,
                                       IntArrayRef stride, IntArrayRef padding,
                                       IntArrayRef dilation, bool transposed,
                                       IntArrayRef output_padding,
                                       int64_t groups);
#endif

void initCUDNNConvolutionBindings(torch::Library &m);

} // namespace operators
} // namespace sfast
