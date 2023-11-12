#pragma once
#include <torch/extension.h>

#include <torch/library.h>

namespace sfast {
namespace operators {

using namespace torch;

Tensor linear_relu(const Tensor &input_t, const Tensor &weight_t,
                   const c10::optional<Tensor> &bias_t = {});

Tensor linear_gelu(const Tensor &input_t, const Tensor &weight_t,
                   const c10::optional<Tensor> &bias_t = {});

void initFusedLinearBindings(torch::Library &m);

} // namespace operators
} // namespace sfast
