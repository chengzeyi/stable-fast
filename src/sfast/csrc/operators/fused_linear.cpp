#include <torch/extension.h>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/native/Resize.h>
#include <c10/core/DispatchKey.h>
#include <c10/macros/Macros.h>
#include <c10/util/MaybeOwned.h>
#include <c10/util/irange.h>
#include <torch/library.h>
#include "fused_linear.h"

namespace sfast {
namespace operators {

using namespace torch;

static Tensor linear_activation(const Tensor &input, const Tensor &weight,
                                const c10::optional<Tensor> &bias_opt,
                                bool use_gelu) {
  // See [Note: hacky wrapper removal for optional tensor]
  auto bias = bias_opt.has_value()
                  ? c10::MaybeOwned<Tensor>::borrowed(*bias_opt)
                  : c10::MaybeOwned<Tensor>::owned(c10::in_place);
  if (input.dim() == 2 && bias->defined()) {
    // Fused op is marginally faster.
    return at::_addmm_activation(*bias, input, weight.t(), 1, 1, use_gelu);
  }
  if (input.dim() == 3 && bias->defined() && input.is_contiguous()) {
    // Also hit the fused path for contiguous 3D input.
    const auto input_sizes = input.sizes();
    const auto result = at::_addmm_activation(
        *bias, input.view({input_sizes[0] * input_sizes[1], input_sizes[2]}),
        weight.t(), 1, 1, use_gelu);
    return result.view({input_sizes[0], input_sizes[1], result.size(1)});
  }
  auto output = at::linear(input, weight, *bias);
  if (use_gelu) {
    output = at::gelu_(output);
  } else {
    output = at::relu_(output);
  }
  return output;
}

Tensor linear_relu(const Tensor &input, const Tensor &weight,
                   const c10::optional<Tensor> &bias) {
  return linear_activation(input, weight, bias, false);
}

Tensor linear_gelu(const Tensor &input, const Tensor &weight,
                   const c10::optional<Tensor> &bias) {
  return linear_activation(input, weight, bias, true);
}

void initFusedLinearBindings(torch::Library &m) {
  m.def("linear_relu",
        dispatch(c10::DispatchKey::CompositeExplicitAutograd, linear_relu));
  m.def("linear_gelu",
        dispatch(c10::DispatchKey::CompositeExplicitAutograd, linear_gelu));
}

} // namespace operators
} // namespace sfast
