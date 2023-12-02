#pragma once
#include <torch/extension.h>

namespace sfast {
namespace operators {
torch::Tensor cutlass_linear_geglu(const torch::Tensor &input,
                                   const torch::Tensor &weight0,
                                   const c10::optional<torch::Tensor> &bias0,
                                   const torch::Tensor &weight1,
                                   const c10::optional<torch::Tensor> &bias1);

torch::Tensor
cutlass_linear_geglu_unified(const torch::Tensor &input,
                             const torch::Tensor &weight,
                             const c10::optional<torch::Tensor> &bias);
} // namespace operators
} // namespace sfast
