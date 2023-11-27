#pragma once
/*
This file contains some of the auxiliary functions used by both Conv.cpp &
Linear.cpp (introduced in a later PR)
*/
#include <torch/extension.h>

// #include <ATen/native/cudnn/Macros.h>

#include <ATen/Tensor.h>
#include <ATen/native/quantized/PackedParams.h>
#include <c10/core/QScheme.h>
#include <c10/util/ArrayRef.h>

namespace sfast {
namespace operators {

struct PackedLinearWeightCutlass : public LinearPackedParamsBase {
  PackedLinearWeightCutlass(at::Tensor orig_weight,
                            c10::optional<at::Tensor> bias,
                            c10::QScheme q_scheme)
      : orig_weight(std::move(orig_weight)), bias_(std::move(bias)),
        q_scheme(std::move(q_scheme)) {}

  at::Tensor apply(at::Tensor input, double output_scale,
                   int64_t output_zero_point) override {
    throw std::runtime_error("apply is not implemented for this packed "
                             "parameter type");
  }
  at::Tensor apply_relu(at::Tensor input, double output_scale,
                        int64_t output_zero_point) override {
    throw std::runtime_error("apply_relu is not implemented for this packed "
                             "parameter type");
  }

  at::Tensor apply_dynamic(at::Tensor input,
                           bool reduce_range = false) override;
  at::Tensor apply_dynamic_relu(at::Tensor input,
                                bool reduce_range = false) override {
    throw std::runtime_error("apply_dynamic_relu is not implemented for this "
                             "packed parameter type");
  }

  std::tuple<at::Tensor, c10::optional<at::Tensor>> unpack() override;

  c10::optional<at::Tensor> bias() override { return bias_; }

  static c10::intrusive_ptr<LinearPackedParamsBase>
  prepack(at::Tensor weight, c10::optional<at::Tensor> bias);

  static c10::intrusive_ptr<LinearPackedParamsBase>
  from_native(const c10::intrusive_ptr<LinearPackedParamsBase> &other);

private:
  at::Tensor orig_weight;
  c10::optional<at::Tensor> bias_;
  c10::QScheme q_scheme;
};

} // namespace operators
} // namespace sfast
