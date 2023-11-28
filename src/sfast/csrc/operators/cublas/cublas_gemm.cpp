#include <torch/extension.h>

#include <c10/core/DispatchKey.h>
#include <torch/library.h>

#if defined(WITH_CUDA)
// #define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/OpMathType.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/NamedTensor.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/Resize.h>
#include <c10/util/MaybeOwned.h>

// #ifndef AT_PER_OPERATOR_HEADERS
// #include <ATen/Functions.h>
// #include <ATen/NativeFunctions.h>
// #else
// #include <ATen/ops/_addmm_activation_native.h>
// #include <ATen/ops/_efficientzerotensor.h>
// #include <ATen/ops/addmm_native.h>
// #include <ATen/ops/addmv_native.h>
// #include <ATen/ops/baddbmm_native.h>
// #include <ATen/ops/bmm_native.h>
// #include <ATen/ops/copy_native.h>
// #include <ATen/ops/dot_native.h>
// #include <ATen/ops/empty.h>
// #include <ATen/ops/gelu.h>
// #include <ATen/ops/mm_native.h>
// #include <ATen/ops/mul.h>
// #include <ATen/ops/relu.h>
// #include <ATen/ops/scalar_tensor_native.h>
// #include <ATen/ops/vdot_native.h>
// #endif

#include "CUDABlas.h"
#include "cublas_gemm.h"

namespace sfast {
namespace operators {

using namespace at;
using namespace at::native;

namespace {

// TODO:
// https://github.com/pytorch/pytorch/pull/59380#pullrequestreview-725310492
c10::MaybeOwned<Tensor> inline resolve_conj_if_indicated(const Tensor &tensor,
                                                         bool resolve_conj) {
  if (resolve_conj && tensor.is_conj()) {
    return c10::MaybeOwned<Tensor>::owned(tensor.resolve_conj());
  } else {
    return c10::MaybeOwned<Tensor>::borrowed(tensor);
  }
}

c10::MaybeOwned<Tensor> inline prepare_matrix_for_cublas(
    const Tensor &tensor, bool &transpose_tensor, bool transpose_result) {
  if (tensor.is_non_overlapping_and_dense()) { // common case
    transpose_tensor = tensor.is_contiguous();
    return resolve_conj_if_indicated(
        tensor, transpose_result ? transpose_tensor : !transpose_tensor);
  }
  IntArrayRef tensor_strides = tensor.strides();
  IntArrayRef tensor_sizes = tensor.sizes();
  if ((tensor_strides[0] == 1) &&
      (tensor_strides[1] >= std::max<int64_t>(1, tensor_sizes[0]))) {
    transpose_tensor = false;
    return resolve_conj_if_indicated(tensor, !transpose_result);
  } else if ((tensor_strides[1] == 1) &&
             (tensor_strides[0] >= std::max<int64_t>(1, tensor_sizes[1]))) {
    transpose_tensor = true;
    return resolve_conj_if_indicated(tensor, transpose_result);
  } else {
    transpose_tensor = true;
    return c10::MaybeOwned<Tensor>::owned(
        tensor.clone(at::MemoryFormat::Contiguous));
  }
}

c10::MaybeOwned<Tensor> inline prepare_matrix_for_cublas(
    const Tensor &tensor, bool &transpose_tensor) {
  if (tensor.is_non_overlapping_and_dense()) { // common case
    transpose_tensor = tensor.is_contiguous();
    return resolve_conj_if_indicated(tensor, true);
  }
  IntArrayRef tensor_strides = tensor.strides();
  IntArrayRef tensor_sizes = tensor.sizes();
  if ((tensor_strides[0] == 1) &&
      (tensor_strides[1] >= std::max<int64_t>(1, tensor_sizes[0]))) {
    transpose_tensor = false;
    return resolve_conj_if_indicated(tensor, true);
  } else if ((tensor_strides[1] == 1) &&
             (tensor_strides[0] >= std::max<int64_t>(1, tensor_sizes[1]))) {
    transpose_tensor = true;
    return resolve_conj_if_indicated(tensor, true);
  } else {
    transpose_tensor = true;
    return c10::MaybeOwned<Tensor>::owned(
        tensor.clone(at::MemoryFormat::Contiguous));
  }
}

c10::MaybeOwned<Tensor> prepare_batch_matrix_for_cublas(const Tensor &tensor,
                                                        bool &transpose_tensor,
                                                        int64_t &ld_tensor,
                                                        bool transpose_result,
                                                        int64_t m, int64_t n) {
  IntArrayRef tensor_strides = tensor.strides();
  c10::MaybeOwned<Tensor> tensor_;
  int fast_dim = transpose_result ? 2 : 1;
  int leading_dim = transpose_result ? 1 : 2;

  if (tensor_strides[fast_dim] == 1 &&
      (tensor_strides[leading_dim] >= std::max<int64_t>(1, m))) {
    transpose_tensor = false;
    tensor_ = resolve_conj_if_indicated(tensor, true);
    ld_tensor = tensor_->strides()[leading_dim];
  } else if ((tensor_strides[leading_dim] == 1) &&
             (tensor_strides[fast_dim] >= std::max<int64_t>(1, n))) {
    transpose_tensor = true;
    tensor_ = resolve_conj_if_indicated(tensor, false);
    ld_tensor = tensor_->strides()[fast_dim];
  } else {
    transpose_tensor = !transpose_result;
    // gemm call requires leading dimension and stride parameters to be non-zero
    bool is_stride_non_zero =
        tensor.strides()[1] != 0 && tensor.strides()[2] != 0;
    if (tensor.is_contiguous() && is_stride_non_zero) {
      tensor_ = resolve_conj_if_indicated(tensor, transpose_result);
    } else {
      tensor_ = c10::MaybeOwned<Tensor>::owned(
          tensor.clone(at::MemoryFormat::Contiguous));
    }
    ld_tensor = tensor_->strides()[1];
  }

  return tensor_;
}

} // namespace

namespace {

enum class Activation {
  None,
  RELU,
  GELU,
};

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000 && !defined(_MSC_VER)
operators::blas::GEMMAndBiasActivationEpilogue
activation_to_gemm_and_blas_arg(Activation a) {
  switch (a) {
  case Activation::None:
    return operators::blas::GEMMAndBiasActivationEpilogue::None;
  case Activation::RELU:
    return operators::blas::GEMMAndBiasActivationEpilogue::RELU;
  case Activation::GELU:
    return operators::blas::GEMMAndBiasActivationEpilogue::GELU;
  default:
    TORCH_CHECK(false);
    return operators::blas::GEMMAndBiasActivationEpilogue::None;
  }
}
#endif

Tensor &addmm_out_cuda_impl(Tensor &result, const Tensor &self,
                            const Tensor &mat1, const Tensor &mat2,
                            const Scalar &beta, const Scalar &alpha,
                            Activation activation = Activation::None,
                            const c10::optional<Tensor> &other = {},
                            const Scalar &gamma = 1.0) {
  // Make sure to keep addmm_cuda below in sync with this code; it
  // preflights a check to try to avoid actually needing to call
  // expand().
  TORCH_CHECK(mat1.dim() == 2 && mat2.dim() == 2, "tensors must be 2-D");

  TensorArg args[]{{result, "out", 0},
                   {self, "self", 1},
                   {mat1, "mat1", 2},
                   {mat2, "mat2", 3}};
  checkAllSameGPU(__func__, args);

  IntArrayRef mat1_sizes = mat1.sizes();
  IntArrayRef mat2_sizes = mat2.sizes();
  IntArrayRef self__sizes;
  bool useLtInterface = false;
  at::ScalarType scalar_type = self.scalar_type();
  c10::MaybeOwned<Tensor> self_;
  if (&result != &self) {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11030 && !defined(_MSC_VER)
    // Strangely, if mat2 has only 1 row or column, we get
    // CUBLAS_STATUS_INVALID_VALUE error from cublasLtMatmulAlgoGetHeuristic.
    // self.dim() == 1 && result.dim() == 2 && self.sizes()[0] == mat2_sizes[1]
    // is to use lt interface only when self is bias.
    // for cuda 11.4, cublasLtMatmul is activated
    // the last two conditions is to skip 16b transA and non-trans-B having
    // leading dim >> rows when they are sliced from a large tensor
    // see fbcode/caffe2/test/test_linalg.py:test_corner_cases_of_cublasltmatmul
    useLtInterface =
        beta.toComplexDouble() == 1.0 && self.dim() == 1 && result.dim() == 2 &&
        self.sizes()[0] == mat2_sizes[1] && self.is_contiguous() &&
        result.is_contiguous() &&
        (scalar_type == at::ScalarType::Double ||
         scalar_type == at::ScalarType::Float ||
         scalar_type == at::ScalarType::Half ||
         scalar_type == at::ScalarType::BFloat16) &&
        mat2_sizes[0] > 1 && mat2_sizes[1] > 1 && mat2_sizes[0] < 65535 * 32 &&
        mat2_sizes[1] < 65535 * 32 && mat1_sizes[0] < 65535 * 32 &&
        mat1_sizes[1] < 65535 * 32 &&
        // avoid leaing dim >> rows bugs
        ((mat1.strides()[0] == 1 && mat1.strides()[1] == mat1_sizes[0]) ||
         (mat1.strides()[1] == 1 && mat1.strides()[0] == mat1_sizes[1]) ||
         (scalar_type != at::ScalarType::Half &&
          scalar_type != at::ScalarType::BFloat16)) &&
        ((mat2.strides()[0] == 1 && mat2.strides()[1] == mat2_sizes[0]) ||
         (mat2.strides()[1] == 1 && mat2.strides()[0] == mat2_sizes[1]) ||
         (scalar_type != at::ScalarType::Half &&
          scalar_type != at::ScalarType::BFloat16));
#endif
    if (!useLtInterface) {
      self_ = expand_size(self, {mat1_sizes[0], mat2_sizes[1]}, "addmm");
    }
    self__sizes = self_->sizes();
  } else {
    self_ = c10::MaybeOwned<Tensor>::borrowed(self);
    self__sizes = self_->sizes();
    TORCH_CHECK(result.dim() == 2, "tensors must be 2-D");
    TORCH_CHECK(self__sizes[0] == mat1_sizes[0],
                "self_ dim 0 must match mat1 dim 0");
    TORCH_CHECK(self__sizes[1] == mat2_sizes[1],
                "self_ dim 1 must match mat2 dim 1");
  }

  if (&result != &self) {
    at::native::resize_output(result, {mat1_sizes[0], mat2_sizes[1]});
    if (beta.toComplexDouble() != 0.0 && !useLtInterface) {
      at::native::copy_(result, *self_);
    }
  }

  IntArrayRef result_sizes = result.sizes();
  if ((result_sizes[0] == 0) || (result_sizes[1] == 0)) {
    return result;
  }

  bool transpose_result;
  c10::MaybeOwned<Tensor> result_ =
      prepare_matrix_for_cublas(result, transpose_result);
  bool transpose_mat1;
  bool transpose_mat2;
  auto mat1_ = prepare_matrix_for_cublas(transpose_result ? mat2 : mat1,
                                         transpose_mat1, transpose_result);
  auto mat2_ = prepare_matrix_for_cublas(transpose_result ? mat1 : mat2,
                                         transpose_mat2, transpose_result);
  c10::MaybeOwned<Tensor> other_;
  Scalar gamma_;
  if (other.has_value() && gamma.toComplexDouble() != 0.0) {
    other_ = prepare_matrix_for_cublas(other.value(), transpose_result);
    gamma_ = gamma;
  } else {
    other_ = result_;
    gamma_ = 0;
  }

  if (transpose_result) {
    transpose_mat1 = !transpose_mat1;
    transpose_mat2 = !transpose_mat2;
    mat1_sizes = mat1_->sizes();
    mat2_sizes = mat2_->sizes();
  }

  int64_t m = mat1_sizes[transpose_result ? 1 : 0];
  int64_t k = mat1_sizes[transpose_result ? 0 : 1];
  int64_t n = mat2_sizes[transpose_result ? 0 : 1];
  int64_t mat1_ld = mat1_->stride((transpose_mat1 == transpose_result) ? 1 : 0);
  int64_t mat2_ld = mat2_->stride((transpose_mat2 == transpose_result) ? 1 : 0);
  int64_t result_ld = result_->stride(transpose_result ? 0 : 1);
  int64_t other_ld = gamma_.toComplexDouble() == 0.0
                         ? 0
                         : other_->stride(transpose_result ? 0 : 1);

  if (mat1.numel() == 0) {
    // By definition, when beta==0, values in self should be ignored. nans and
    // infs should not propagate
    if (beta.toComplexDouble() == 0.) {
      if (gamma_.toComplexDouble() == 0.) {
        return result.zero_();
      } else {
        return result.zero_().add_(*other_, gamma_);
      }
    }
    // TODO: We could squeeze some perf by calling at::cuda::mul_out here
    // instead, to bypass the dispatcher. That requires some fixing some
    // internal build dependencies though.
    if (gamma_.toComplexDouble() == 0.) {
      return at::mul_out(
          result, self,
          at::native::scalar_tensor(beta, self.scalar_type(),
                                    c10::nullopt /* layout */, at::kCPU,
                                    c10::nullopt /* pin_memory */));
    } else {
      return at::mul_out(
                 result, self,
                 at::native::scalar_tensor(beta, self.scalar_type(),
                                           c10::nullopt /* layout */, at::kCPU,
                                           c10::nullopt /* pin_memory */))
          .add_(*other_, gamma_);
    }
  }

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!result_->is_conj());

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000 && !defined(_MSC_VER)
  if (useLtInterface) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, scalar_type,
        "addmm_cuda_lt", [&] {
          sfast::operators::blas::gemm_and_bias<scalar_t>(
              transpose_mat1, transpose_mat2, m, n, k,
              alpha.to<at::opmath_type<scalar_t>>(),
              mat1_->data_ptr<scalar_t>(), mat1_ld, mat2_->data_ptr<scalar_t>(),
              mat2_ld, self.data_ptr<scalar_t>(), result_->data_ptr<scalar_t>(),
              result_ld,
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11040
              // #if 0
              activation_to_gemm_and_blas_arg(activation),
#else
              // GELU is not supported (and does not compile!) prior
              // to CUDA 11.4.  Have observed accuracy issues with
              // GELU epilogue in 11.4; disabling the GELU epilogue
              // path until we confirm which version it's working in.
              activation != Activation::GELU
              ? activation_to_gemm_and_blas_arg(activation)
              : operators::blas::GEMMAndBiasActivationEpilogue::None,
#endif
              gamma_.to<at::opmath_type<scalar_t>>(),
              other_->data_ptr<scalar_t>(), other_ld);
        });
  } else
#endif
  {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, scalar_type,
        "addmm_cuda", [&] {
          using opmath_t = at::opmath_type<scalar_t>;
          opmath_t alpha_val = alpha.to<opmath_t>();
          opmath_t beta_val = beta.to<opmath_t>();
          scalar_t *mat1_ptr = mat1_->data_ptr<scalar_t>();
          scalar_t *mat2_ptr = mat2_->data_ptr<scalar_t>();
          scalar_t *result_ptr = result_->data_ptr<scalar_t>();
          sfast::operators::blas::gemm<scalar_t>(
              transpose_mat1 ? mat1_->is_conj() ? 'c' : 't' : 'n',
              transpose_mat2 ? mat2_->is_conj() ? 'c' : 't' : 'n', m, n, k,
              alpha_val, mat1_ptr, mat1_ld, mat2_ptr, mat2_ld, beta_val,
              result_ptr, result_ld);
        });
    if (gamma_.toComplexDouble() != 0.) {
      result_->add_(*other_, gamma_);
    }
    switch (activation) {
    case Activation::RELU:
      at::relu_(const_cast<Tensor &>(*result_));
      break;
    case Activation::GELU:
      at::gelu_(const_cast<Tensor &>(*result_));
      break;
    default:
      break;
    }
  }

// Preprocessor gate here needs to match the inverse of the check
// gating activation_to_gemm_and_blas_arg above; here we are manually
// performing a post-GELU because we weren't able to use the GELU
// epilogue above.
#if defined(CUDA_VERSION) && CUDA_VERSION < 11040
  // #if !0
  if (useLtInterface && activation == Activation::GELU) {
    at::gelu_(const_cast<Tensor &>(*result_));
  }
#endif

  if (!result.is_same(*result_)) {
    result.copy_(*result_);
  }
  return result;
}

const Tensor &baddbmm_out_cuda_impl(const Tensor &result, const Tensor &self,
                                    const Tensor &batch1, const Tensor &batch2,
                                    const Scalar &beta, const Scalar &alpha) {
  IntArrayRef batch1_sizes = batch1.sizes();

  // handle pathological cases that blas may not like
  if (result.numel() == 0) {
    return result;
  } else if (batch1_sizes[2] == 0) {
    if (beta.to<c10::complex<double>>() == 0.0) {
      return result.zero_();
    } else {
      return result.mul_(beta);
    }
  }

  bool transpose_result = false;
  c10::MaybeOwned<Tensor> result_;
  IntArrayRef result_strides = result.strides();
  IntArrayRef result_sizes = result.sizes();

  if ((result_strides[1] == 1) &&
      ((result_sizes[2] == 1) ||
       (result_strides[2] >= std::max<int64_t>(1, result_sizes[1])))) {
    result_ = resolve_conj_if_indicated(result, true);
  } else if ((result_strides[2] == 1) &&
             (result_sizes[1] == 1 ||
              (result_strides[1] >= std::max<int64_t>(1, result_sizes[2])))) {
    transpose_result = true;
    result_ = resolve_conj_if_indicated(result, true);
  } else {
    result_ =
        c10::MaybeOwned<Tensor>::owned(result.transpose(1, 2)
                                           .clone(at::MemoryFormat::Contiguous)
                                           .transpose(1, 2));
  }

  int leading_dim = transpose_result ? 1 : 2;

  int64_t m = result_sizes[transpose_result ? 2 : 1];
  int64_t n = result_sizes[leading_dim];
  int64_t k = (transpose_result ? batch2 : batch1).sizes()[leading_dim];

  int64_t lda, ldb, ldc;
  bool transpose_batch1, transpose_batch2;
  auto batch1_ = prepare_batch_matrix_for_cublas(
      transpose_result ? batch2 : batch1, transpose_batch1, lda,
      transpose_result, m, k);
  auto batch2_ = prepare_batch_matrix_for_cublas(
      transpose_result ? batch1 : batch2, transpose_batch2, ldb,
      transpose_result, k, n);

  ldc = result_->strides()[leading_dim];
  int64_t num_batches = result_->sizes()[0];

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!result_->is_conj());

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, self.scalar_type(),
      "baddbmm_cuda", [&] {
        using opmath_t = at::opmath_type<scalar_t>;
        opmath_t alpha_val = alpha.to<opmath_t>();
        opmath_t beta_val = beta.to<opmath_t>();
        scalar_t *batch1_ptr = batch1_->data_ptr<scalar_t>();
        scalar_t *batch2_ptr = batch2_->data_ptr<scalar_t>();
        scalar_t *result_ptr = result_->data_ptr<scalar_t>();
        sfast::operators::blas::bgemm<scalar_t>(
            transpose_batch1 ? batch1_->is_conj() ? 'c' : 't' : 'n',
            transpose_batch2 ? batch2_->is_conj() ? 'c' : 't' : 'n', m, n, k,
            alpha_val, batch1_ptr, lda, batch1_->strides()[0], batch2_ptr, ldb,
            batch2_->strides()[0], beta_val, result_ptr, ldc,
            result_->strides()[0], num_batches);
      });
  if (!result.is_same(*result_)) {
    result.copy_(*result_);
  }
  return result;
}

} // anonymous namespace

/*
TORCH_IMPL_FUNC(addmm_out_cuda)(const Tensor& self, const Tensor& mat1, const
Tensor& mat2, const Scalar& beta, const Scalar& alpha, const Tensor& result) {
  addmm_out_cuda_impl(const_cast<Tensor&>(result), self, mat1, mat2, beta,
alpha);
}

TORCH_IMPL_FUNC(addmm_activation_out_cuda)(const Tensor& self, const Tensor&
mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha, bool
use_gelu, const Tensor& result) {
  addmm_out_cuda_impl(const_cast<Tensor&>(result), self, mat1, mat2, beta,
alpha, use_gelu ? Activation::GELU : Activation::RELU);
}

TORCH_IMPL_FUNC(mm_out_cuda)(const Tensor& self, const Tensor& mat2, const
Tensor& result) { addmm_out_cuda_impl(const_cast<Tensor&>(result), result, self,
mat2, 0, 1);
}

TORCH_IMPL_FUNC(baddbmm_out_cuda)(const Tensor& self, const Tensor& batch1,
const Tensor& batch2, const Scalar& beta, const Scalar& alpha, const Tensor&
result) {
  {
    at::NoNamesGuard guard;
    baddbmm_out_cuda_impl(result, self, batch1, batch2, beta, alpha);
  }
}

TORCH_IMPL_FUNC(bmm_out_cuda)(const Tensor& batch1, const Tensor& batch2, const
Tensor &result) { Scalar beta(0.0); Scalar alpha(1.0);
  {
    NoNamesGuard guard;
    baddbmm_out_cuda_impl(result, result, batch1, batch2, beta, alpha);
  }
}

*/

namespace {

inline void dot_check(const Tensor &self, const Tensor &other) {
  TORCH_CHECK(self.dim() == 1 && other.dim() == 1,
              "1D tensors expected, but got ", self.dim(), "D and ",
              other.dim(), "D tensors");
  TORCH_CHECK(self.scalar_type() == other.scalar_type(),
              "dot : expected both vectors to have same dtype, but found ",
              self.scalar_type(), " and ", other.scalar_type());
  TORCH_CHECK(self.numel() == other.numel(),
              "inconsistent tensor size, expected tensor [", self.numel(),
              "] and src [", other.numel(),
              "] to have the same number of elements, but got ", self.numel(),
              " and ", other.numel(), " elements respectively");
  TORCH_CHECK(self.device() == other.device(),
              "expected all tensors to be on the same device. Found: ",
              self.device(), ", ", other.device());
  TORCH_CHECK((self.numel() <= INT_MAX) && (self.stride(0) <= INT_MAX) &&
                  (other.stride(0) <= INT_MAX),
              "dot only supports n, incx, incy with the bound [val] <= %d",
              INT_MAX);
}

} // anonymous namespace

Tensor dot_cuda(const Tensor &self, const Tensor &other) {
  if (self.is_complex()) {
    if (self.is_conj()) {
      if (other.is_conj()) {
        return (dot_cuda(self.conj(), other.conj())).conj();
      } else {
        return vdot_cuda(self.conj(), other);
      }
    } else if (other.is_conj()) {
      return vdot_cuda(other.conj(), self);
    }
  }

  at::NoNamesGuard guard;
  dot_check(self, other);

  const int n = static_cast<int>(self.numel());
  int incx = static_cast<int>(self.stride(0));
  int incy = static_cast<int>(other.stride(0));
  if (n == 1) {
    incx = 1;
    incy = 1;
  }

  if (self._is_zerotensor() || other._is_zerotensor()) {
    return at::_efficientzerotensor({}, self.options());
  }

  return AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      ScalarType::Half, ScalarType::BFloat16, self.scalar_type(), "dot", [&] {
        Tensor result = at::empty({}, self.options());

        auto handle = at::cuda::getCurrentCUDABlasHandle();
        sfast::operators::blas::PointerModeGuard pointerModeGuard(
            handle, CUBLAS_POINTER_MODE_DEVICE);
        sfast::operators::blas::dot<scalar_t>(
            handle, n, self.data_ptr<scalar_t>(), incx,
            other.data_ptr<scalar_t>(), incy, result.data_ptr<scalar_t>());

        return result;
      });
}

Tensor vdot_cuda(const Tensor &self, const Tensor &other) {
  if (!self.is_complex()) {
    return dot_cuda(self, other);
  }

  if (self.is_conj()) {
    if (other.is_conj()) {
      return vdot_cuda(other.conj(), self.conj());
    } else {
      return dot_cuda(self.conj(), other);
    }
  } else if (other.is_conj()) {
    return (dot_cuda(self, other.conj())).conj();
  }

  at::NoNamesGuard guard;
  dot_check(self, other);

  if (self._is_zerotensor() || other._is_zerotensor()) {
    return at::_efficientzerotensor({}, self.options());
  }

  const int n = static_cast<int>(self.numel());
  int incx = static_cast<int>(self.stride(0));
  int incy = static_cast<int>(other.stride(0));
  if (n == 1) {
    incx = 1;
    incy = 1;
  }

  return AT_DISPATCH_COMPLEX_TYPES(self.scalar_type(), "vdot", [&] {
    Tensor result = at::empty({}, self.options());

    auto handle = at::cuda::getCurrentCUDABlasHandle();
    sfast::operators::blas::PointerModeGuard pointerModeGuard(
        handle, CUBLAS_POINTER_MODE_DEVICE);
    sfast::operators::blas::vdot<scalar_t>(handle, n, self.data_ptr<scalar_t>(),
                                           incx, other.data_ptr<scalar_t>(),
                                           incy, result.data_ptr<scalar_t>());

    return result;
  });
}

/*
TORCH_IMPL_FUNC(addmv_out_cuda)(const Tensor &self, const Tensor &mat, const
Tensor &vec, const Scalar& beta_, const Scalar& alpha_, const Tensor& result) {
  c10::MaybeOwned<Tensor> self_ = expand_size(self, {mat.size(0)});
  auto betaval = beta_.toComplexDouble();
  if (mat.numel() == 0) {
    // shortcut for an empty matrix
    // By definition, when beta==0, values in self should be ignored. nans and
infs
    // should not propagate
    if (betaval == 0.0) {
      result.zero_();
    } else {
      at::mul_out(
          const_cast<Tensor&>(result),
          self,
          at::native::scalar_tensor(
              beta_, self.scalar_type(), c10::nullopt /* layout */
/*, at::kCPU, c10::nullopt /* pin_memory */ /*));
}
} else {
if (!result.is_same(*self_) && betaval != 0.0) { //if beta is 0, result contents
will be zeroed later at::native::copy_(const_cast<Tensor&>(result), *self_);
}
if (result.numel() != 0) {
auto r_stride = result.stride(0);
auto vec_stride = vec.stride(0);

// Check for contiguity of `vec` and update `vec_stride` accordingly
const auto vec_contiguous = vec_stride == 0 ? vec.contiguous() : vec;
// A vector can be contiguous and have a stride of zero if it has it is of
length 1 vec_stride = std::max<int64_t>(vec_contiguous.stride(0), 1LL);

AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(at::ScalarType::Half,
at::ScalarType::BFloat16, mat.scalar_type(), "addmv_impl_cuda", [&] { auto beta
= beta_.to<scalar_t>(); auto alpha = alpha_.to<scalar_t>(); if (mat.stride(0) ==
1 && mat.stride(1) >= std::max<int64_t>(1, mat.size(0))) {
at::cuda::blas::gemv<scalar_t>('n',
mat.size(0), mat.size(1), alpha, mat.data_ptr<scalar_t>(), mat.stride(1),
vec_contiguous.data_ptr<scalar_t>(), vec_stride, beta,
result.data_ptr<scalar_t>(), r_stride);
}
else if (mat.stride(1) == 1 && mat.stride(0) >= std::max<int64_t>(1,
mat.size(1))) { at::cuda::blas::gemv<scalar_t>('t', mat.size(1), mat.size(0),
alpha, mat.data_ptr<scalar_t>(), mat.stride(0),
vec_contiguous.data_ptr<scalar_t>(), vec_stride, beta,
result.data_ptr<scalar_t>(), r_stride);
}
else {
Tensor cmat = mat.contiguous();
at::cuda::blas::gemv<scalar_t>('t',
mat.size(1), mat.size(0), alpha, cmat.data_ptr<scalar_t>(), cmat.stride(0),
vec_contiguous.data_ptr<scalar_t>(), vec_stride, beta,
result.data_ptr<scalar_t>(), r_stride);
}
});
}
}
}
*/

} // namespace operators
} // namespace sfast
#endif

namespace sfast {
namespace operators {

using namespace torch;

#if defined(WITH_CUDA)
Tensor cublas_lowp_addmm(const Tensor &self, const Tensor &mat1,
                         const Tensor &mat2, const Scalar &beta,
                         const Scalar &alpha) {
  if (self.is_cuda()) {
    auto result = at::empty({mat1.size(0), mat2.size(1)}, self.options());
    return addmm_out_cuda_impl(result, self, mat1, mat2, beta, alpha);
  }
  return at::addmm(self, mat1, mat2, beta, alpha);
}

Tensor cublas_lowp_addmm_add(const Tensor &self, const Tensor &mat1,
                             const Tensor &mat2, const Tensor &other,
                             const Scalar &beta, const Scalar &alpha,
                             const Scalar &gamma) {
  if (self.is_cuda()) {
    auto result = at::empty({mat1.size(0), mat2.size(1)}, self.options());
    return addmm_out_cuda_impl(result, self, mat1, mat2, beta, alpha,
                               Activation::None, other, gamma);
  }
  return at::addmm(self, mat1, mat2, beta, alpha).add_(other, gamma);
}

Tensor cublas_lowp_addmm_activation(const Tensor &self, const Tensor &mat1,
                                    const Tensor &mat2, const Scalar &beta,
                                    const Scalar &alpha, bool use_gelu) {
  if (self.is_cuda()) {
    auto result = at::empty({mat1.size(0), mat2.size(1)}, self.options());
    return addmm_out_cuda_impl(result, self, mat1, mat2, beta, alpha,
                               use_gelu ? Activation::GELU : Activation::RELU);
  }
  return at::_addmm_activation(self, mat1, mat2, beta, alpha);
}

Tensor cublas_lowp_mm(const Tensor &self, const Tensor &mat2) {
  if (self.is_cuda()) {
    auto result = at::empty({self.size(0), mat2.size(1)}, self.options());
    return addmm_out_cuda_impl(result, result, self, mat2, 0, 1);
  }
  return at::mm(self, mat2);
}

Tensor cublas_lowp_baddbmm(const Tensor &self, const Tensor &batch1,
                           const Tensor &batch2, const Scalar &beta,
                           const Scalar &alpha) {
  if (self.is_cuda()) {
    auto result = at::empty({batch1.size(0), batch1.size(1), batch2.size(2)},
                            self.options());
    return baddbmm_out_cuda_impl(result, self, batch1, batch2, beta, alpha);
  }
  return at::baddbmm(self, batch1, batch2, beta, alpha);
}

Tensor cublas_lowp_bmm(const Tensor &self, const Tensor &batch2) {
  if (self.is_cuda()) {
    auto result =
        at::empty({self.size(0), self.size(1), batch2.size(2)}, self.options());
    return baddbmm_out_cuda_impl(result, result, self, batch2, 0, 1);
  }
  return at::bmm(self, batch2);
}

Tensor cublas_lowp_matmul(const Tensor &tensor1, const Tensor &tensor2) {
  if (tensor1.is_cuda()) {
    auto result =
        at::empty({tensor1.size(0), tensor2.size(1)}, tensor1.options());
    const auto dim_tensor1 = tensor1.dim();
    const auto dim_tensor2 = tensor2.dim();

    // This is checked up here to simplify the logic below
    // Note that the strings are just evaluated on failure, so almost always we
    // just evaluate the condition and move on
    TORCH_CHECK(
        dim_tensor1 != 0 && dim_tensor2 != 0,
        "both arguments to matmul need to be at least 1D, but they are ",
        dim_tensor1, "D and ", dim_tensor2, "D");

    if (dim_tensor1 == 1 && dim_tensor2 == 1) {
      return dot_cuda(tensor1, tensor2);
      // } else if (dim_tensor1 == 2 && dim_tensor2 == 1) {
      //   return tensor1.mv(tensor2);
    } else if (dim_tensor1 == 1 && dim_tensor2 == 2) {
      return cublas_lowp_mm(tensor1.unsqueeze(0), tensor2).squeeze_(0);
    } else if (dim_tensor1 == 2 && dim_tensor2 == 2) {
      return cublas_lowp_mm(tensor1, tensor2);
    }
  }

  return at::matmul(tensor1, tensor2);
}

Tensor cublas_lowp_linear(const Tensor &input, const Tensor &weight,
                          const c10::optional<Tensor> &bias_opt) {
  // See [Note: hacky wrapper removal for optional tensor]
  auto bias = bias_opt.has_value()
                  ? c10::MaybeOwned<Tensor>::borrowed(*bias_opt)
                  : c10::MaybeOwned<Tensor>::owned(c10::in_place);

  if (input.is_cuda()) {
    if (input.dim() == 2) {
      if (bias->defined()) {
        // Fused op is marginally faster.
        return cublas_lowp_addmm(*bias, input, weight.t());
      } else {
        return cublas_lowp_mm(input, weight.t());
      }
    } else if (input.dim() == 3 && input.is_contiguous()) {
      // Also hit the fused path for contiguous 3D input.
      const auto input_sizes = input.sizes();
      Tensor result;
      if (bias->defined()) {
        // Fused op is marginally faster.
        result = cublas_lowp_addmm(
            *bias,
            input.view({input_sizes[0] * input_sizes[1], input_sizes[2]}),
            weight.t());
      } else {
        result = cublas_lowp_mm(
            input.view({input_sizes[0] * input_sizes[1], input_sizes[2]}),
            weight.t());
      }
      return result.view({input_sizes[0], input_sizes[1], result.size(1)});
    }
    const auto dim_tensor1 = input.dim();
    const auto dim_tensor2 = weight.dim();
    if (bias->defined()) {
      if (dim_tensor1 == 1 && dim_tensor2 == 2) {
        return cublas_lowp_addmm(*bias, input.unsqueeze(0), weight.t())
            .squeeze_(0);
      } else if (dim_tensor1 == 2 && dim_tensor2 == 2) {
        return cublas_lowp_addmm(*bias, input, weight.t());
      }
    }
    auto output = cublas_lowp_matmul(input, weight.t());
    if (bias->defined()) {
      output.add_(*bias);
      // for composite compliance use out-of-place version of `add`
      // if (isTensorSubclassLike(*bias)) {
      //   output = at::add(output, *bias);
      // } else {
      //   output.add_(*bias);
      // }
    }
    return output;
  }
  return at::linear(input, weight, bias_opt);
}

Tensor cublas_lowp_linear_activation(const Tensor &input, const Tensor &weight,
                                     const c10::optional<Tensor> &bias_opt,
                                     bool use_gelu) {
  // See [Note: hacky wrapper removal for optional tensor]
  auto bias = bias_opt.has_value()
                  ? c10::MaybeOwned<Tensor>::borrowed(*bias_opt)
                  : c10::MaybeOwned<Tensor>::owned(c10::in_place);
  if (input.dim() == 2 && bias->defined()) {
    // Fused op is marginally faster.
    return cublas_lowp_addmm_activation(*bias, input, weight.t(), 1, 1,
                                        use_gelu);
  }
  if (input.dim() == 3) {
    // Also hit the fused path for contiguous 3D input.
    const auto input_sizes = input.sizes();
    if (bias->defined() &&
        at::detail::computeStride(
            input.sizes(), input.strides(),
            IntArrayRef({input_sizes[0] * input_sizes[1], input_sizes[2]}))
            .has_value()) {
      const auto result = cublas_lowp_addmm_activation(
          *bias, input.view({input_sizes[0] * input_sizes[1], input_sizes[2]}),
          weight.t(), 1, 1, use_gelu);
      return result.view({input_sizes[0], input_sizes[1], result.size(1)});
    }
  }
  auto output = cublas_lowp_linear(input, weight, *bias);
  if (use_gelu) {
    output = at::gelu_(output);
  } else {
    output = at::relu_(output);
  }
  return output;
}

Tensor cublas_lowp_linear_relu(const Tensor &input, const Tensor &weight,
                               const c10::optional<Tensor> &bias_opt) {
  return cublas_lowp_linear_activation(input, weight, bias_opt, false);
}

Tensor cublas_lowp_linear_gelu(const Tensor &input, const Tensor &weight,
                               const c10::optional<Tensor> &bias_opt) {
  return cublas_lowp_linear_activation(input, weight, bias_opt, true);
}

Tensor cublas_lowp_linear_add(const Tensor &input, const Tensor &weight,
                              const c10::optional<Tensor> &bias_opt,
                              const Tensor &other, const Scalar &alpha) {
  // See [Note: hacky wrapper removal for optional tensor]
  auto bias = bias_opt.has_value()
                  ? c10::MaybeOwned<Tensor>::borrowed(*bias_opt)
                  : c10::MaybeOwned<Tensor>::owned(c10::in_place);

  if (input.is_cuda()) {
    if (input.dim() == 2) {
      if (bias->defined()) {
        // Fused op is marginally faster.
        return cublas_lowp_addmm_add(*bias, input, weight.t(), other, 1, 1,
                                     alpha);
      }
    } else if (input.dim() == 3) {
      // Also hit the fused path for 3D input.
      const auto input_sizes = input.sizes();
      if (bias->defined() &&
          at::detail::computeStride(
              input.sizes(), input.strides(),
              IntArrayRef({input_sizes[0] * input_sizes[1], input_sizes[2]}))
              .has_value() &&
          at::detail::computeStride(
              other.sizes(), other.strides(),
              IntArrayRef({input_sizes[0] * input_sizes[1], weight.size(0)}))
              .has_value()) {
        // Fused op is marginally faster.
        auto result = cublas_lowp_addmm_add(
            *bias,
            input.view({input_sizes[0] * input_sizes[1], input_sizes[2]}),
            weight.t(),
            other.view({input_sizes[0] * input_sizes[1], weight.size(0)}), 1, 1,
            alpha);
        return result.view({input_sizes[0], input_sizes[1], result.size(1)});
      }
    }
    const auto dim_tensor1 = input.dim();
    const auto dim_tensor2 = weight.dim();
    if (bias->defined()) {
      if (dim_tensor1 == 1 && dim_tensor2 == 2) {
        return cublas_lowp_addmm_add(*bias, input.unsqueeze(0), weight.t(),
                                     other.unsqueeze(0), 1, 1, alpha)
            .squeeze_(0);
      }
    }
  }
  return cublas_lowp_linear(input, weight, bias_opt).add_(other, alpha);
}
#endif

void initCUBLASGEMMBindings(torch::Library &m) {
#if defined(WITH_CUDA)
  m.def(
      "cublas_lowp_addmm",
      dispatch(c10::DispatchKey::CompositeExplicitAutograd, cublas_lowp_addmm));
  m.def("cublas_lowp_addmm_add",
        dispatch(c10::DispatchKey::CompositeExplicitAutograd,
                 cublas_lowp_addmm_add));
  m.def("cublas_lowp_addmm_activation",
        dispatch(c10::DispatchKey::CompositeExplicitAutograd,
                 cublas_lowp_addmm_activation));
  m.def("cublas_lowp_mm",
        dispatch(c10::DispatchKey::CompositeExplicitAutograd, cublas_lowp_mm));
  m.def("cublas_lowp_baddbmm",
        dispatch(c10::DispatchKey::CompositeExplicitAutograd,
                 cublas_lowp_baddbmm));
  m.def("cublas_lowp_bmm",
        dispatch(c10::DispatchKey::CompositeExplicitAutograd, cublas_lowp_bmm));
  m.def("cublas_lowp_matmul",
        dispatch(c10::DispatchKey::CompositeExplicitAutograd,
                 cublas_lowp_matmul));
  m.def("cublas_lowp_linear",
        dispatch(c10::DispatchKey::CompositeExplicitAutograd,
                 cublas_lowp_linear));
  m.def("cublas_lowp_linear_relu",
        dispatch(c10::DispatchKey::CompositeExplicitAutograd,
                 cublas_lowp_linear_relu));
  m.def("cublas_lowp_linear_gelu",
        dispatch(c10::DispatchKey::CompositeExplicitAutograd,
                 cublas_lowp_linear_gelu));
  m.def("cublas_lowp_linear_add",
        dispatch(c10::DispatchKey::CompositeExplicitAutograd,
                 cublas_lowp_linear_add));
#endif
}

} // namespace operators
} // namespace sfast
