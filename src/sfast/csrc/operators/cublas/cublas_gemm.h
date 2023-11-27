#pragma once
#include <torch/extension.h>

#include <torch/library.h>

namespace sfast {
namespace operators {

using namespace torch;

#if defined(WITH_CUDA)
Tensor cublas_lowp_addmm(const Tensor &self, const Tensor &mat1,
                         const Tensor &mat2, const Scalar &beta = 1,
                         const Scalar &alpha = 1);

Tensor cublas_lowp_addmm_add(const Tensor &self, const Tensor &mat1,
                         const Tensor &mat2,
                         const Tensor &other,
                         const Scalar &beta = 1,
                         const Scalar &alpha = 1,
                         const Scalar &gamma = 1);

Tensor cublas_lowp_addmm_activation(const Tensor &self, const Tensor &mat1,
                                    const Tensor &mat2, const Scalar &beta = 1,
                                    const Scalar &alpha = 1,
                                    bool use_gelu = false);

Tensor cublas_lowp_mm(const Tensor &self, const Tensor &mat2);

Tensor cublas_lowp_baddbmm(const Tensor &self, const Tensor &batch1,
                           const Tensor &batch2, const Scalar &beta,
                           const Scalar &alpha);

Tensor cublas_lowp_bmm(const Tensor &self, const Tensor &batch2);

Tensor cublas_lowp_matmul(const Tensor &tensor1, const Tensor &tensor2);

Tensor cublas_lowp_linear(const Tensor &input, const Tensor &weight,
                          const c10::optional<Tensor> &bias_opt);

Tensor cublas_lowp_linear_relu(const Tensor &input, const Tensor &weight,
                               const c10::optional<Tensor> &bias_opt);

Tensor cublas_lowp_linear_gelu(const Tensor &input, const Tensor &weight,
                               const c10::optional<Tensor> &bias_opt);

Tensor cublas_lowp_linear_add(const Tensor &input, const Tensor &weight,
                              const c10::optional<Tensor> &bias_opt,
                              const Tensor &other, const Scalar &alpha = 1);
#endif

void initCUBLASGEMMBindings(torch::Library &m);

} // namespace operators
} // namespace sfast
