#include <torch/extension.h>

#include "cutlass_quantized_utils.h"
#include "cutlass_qlinear_dynamic_kernel.h"
#include "cutlass_qlinear.h"

namespace sfast {
namespace operators {

at::Tensor PackedLinearWeightCutlass::apply_dynamic(at::Tensor input,
                                                  bool reduce_range) {
  if (reduce_range) {
    TORCH_WARN(
        "Currently CUTLASS QLinear ignores reduce_range when it its set to true");
  }

  return sfast::operators::cutlass_qlinear_dynamic(input, orig_weight, bias_);
}

std::tuple<at::Tensor, c10::optional<at::Tensor>>
PackedLinearWeightCutlass::unpack() {
  return std::tuple<at::Tensor, c10::optional<at::Tensor>>{orig_weight, bias_};
}

c10::intrusive_ptr<LinearPackedParamsBase>
PackedLinearWeightCutlass::prepack(at::Tensor weight,
                                 c10::optional<at::Tensor> bias) {
  TORCH_CHECK(weight.qscheme() == c10::kPerTensorAffine,
              "Unsupported qscheme: ", toString(weight.qscheme()));
  const int output_channels = weight.size(0);
  const auto qtype = weight.qscheme();
  if (bias.has_value()) {
    TORCH_CHECK(bias.value().dim() == 1, "bias should be a vector (1D Tensor)");
    TORCH_CHECK(bias.value().size(0) == output_channels,
                "bias should have K elements: " +
                    std::to_string(output_channels));
  }

  auto ret_ptr =
      c10::make_intrusive<PackedLinearWeightCutlass>(weight, bias, qtype);
  return ret_ptr;
}

c10::intrusive_ptr<LinearPackedParamsBase> PackedLinearWeightCutlass::from_native(
    const c10::intrusive_ptr<LinearPackedParamsBase> &other) {
  if (dynamic_cast<PackedLinearWeightCutlass *>(other.get()) != nullptr) {
    return other;
  }
  auto unpacked = other->unpack();
  auto orig_weight = std::get<0>(unpacked);
  auto orig_bias = std::get<1>(unpacked);
  return PackedLinearWeightCutlass::prepack(orig_weight, orig_bias);
}

namespace {

template <bool kReluFused> class QLinearInt8 final {
public:
  static at::Tensor
  run_dynamic(at::Tensor act,
              const c10::intrusive_ptr<LinearPackedParamsBase> &packed_weight,
              bool reduce_range = false) {
    auto packed_weight_cutlass =
        PackedLinearWeightCutlass::from_native(packed_weight);
    if (kReluFused) {
      throw std::runtime_error("Fused ReLU is not supported for CUDNN QLinear");
    } else {
      return packed_weight_cutlass->apply_dynamic(act);
    }
  }
};

TORCH_LIBRARY_IMPL(quantized, QuantizedCUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::linear_dynamic"),
         QLinearInt8<false>::run_dynamic);
}

TORCH_LIBRARY_IMPL(quantized, CUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::linear_dynamic"),
         QLinearInt8<false>::run_dynamic);
}

} // namespace

void initCutlassQLinearBindings(torch::Library &m) {
  m.def("cutlass_qlinear_dynamic",
        torch::dispatch(c10::DispatchKey::CompositeImplicitAutograd,
                        QLinearInt8<false>::run_dynamic));
}

} // namespace operators
} // namespace sfast
