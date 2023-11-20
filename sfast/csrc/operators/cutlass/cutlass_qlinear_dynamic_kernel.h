#pragma once
#include <torch/extension.h>

namespace sfast {
namespace operators {
torch::Tensor cutlass_qlinear_dynamic(const torch::Tensor &input,
                                      const torch::Tensor &weight,
                                      const c10::optional<torch::Tensor> &bias);
} // namespace operators
} // namespace sfast
