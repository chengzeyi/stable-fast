#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>

#include <cutlass/cutlass.h>

#include <cutlass/arch/arch.h>
#include <cutlass/arch/mma.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/thread/linear_combination_gelu.h>
#include <cutlass/epilogue/thread/linear_combination_with_elementwise.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>

#include <device/dual_gemm.h>

#include "operators/cublas/cublas_gemm.h"
#include "thread/mul.h"

#include "cutlass_dual_linear.h"

namespace sfast {
namespace operators {

namespace {

using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

using MMAOp = cutlass::arch::OpClassTensorOp;
using SimtOp = cutlass::arch::OpClassSimt;

namespace sm80_space {
using SmArch = cutlass::arch::Sm80;
constexpr int NumStages = 4;

template <typename scalar_t, typename acc_t> struct GemmConfig {
  using ElementA = scalar_t;
  using ElementB = scalar_t;
  using ElementOutput = scalar_t;
  using ElementAccumulator = acc_t;
  using ElementComputeEpilogue = scalar_t;

  using ThreadBlockShape = cutlass::gemm::GemmShape<128, 64, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
};
} // namespace sm80_space

using namespace sm80_space;

template <typename config> struct GemmGEGLUWrapper {
  using ElementA = typename config::ElementA;
  using ElementB = typename config::ElementB;
  using ElementOutput = typename config::ElementOutput;
  using ElementAccumulator = typename config::ElementAccumulator;
  using ElementComputeEpilogue = typename config::ElementComputeEpilogue;

  using ThreadBlockShape = typename config::ThreadBlockShape;
  using WarpShape = typename config::WarpShape;
  using InstructionShape = typename config::InstructionShape;

  static constexpr const auto kScaleType =
      cutlass::epilogue::thread::ScaleType::NoBetaScaling;

  using EpilogueOutputOp0 = cutlass::epilogue::thread::LinearCombination<
      ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator, ElementComputeEpilogue, kScaleType>;
  using EpilogueOutputOp1 = cutlass::epilogue::thread::LinearCombinationGELU<
      ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator, ElementComputeEpilogue, kScaleType>;
  using EpilogueOutputOp2 = cutlass::epilogue::thread::Mul<
      ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementOutput, ElementComputeEpilogue>;

  using Gemm = cutlass::gemm::device::DualGemm<
      ElementA, LayoutInputA, ElementB, LayoutInputB, LayoutInputB,
      ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch,
      ThreadBlockShape, WarpShape, InstructionShape, EpilogueOutputOp0,
      EpilogueOutputOp1, EpilogueOutputOp2,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, NumStages,
      false, false, false, 128 / cutlass::sizeof_bits<ElementA>::value,
      128 / cutlass::sizeof_bits<ElementB>::value>;
};

void get_input_layout(const torch::Tensor &input, const torch::Tensor &weight0,
                      const torch::Tensor &weight1, int &B, int &M, int &K,
                      int &N, cutlass::MatrixCoord &input_size,
                      cutlass::MatrixCoord &weight_size,
                      cutlass::MatrixCoord &output_size) {
  if (input.dim() == 3) {
    B = input.size(0);
    M = input.size(1);
    K = input.size(2);
  } else {
    B = 1;
    M = input.size(0);
    K = input.size(1);
  }
  // weight is NK
  N = weight0.size(0);
  TORCH_CHECK(weight0.size(1) == K, "weight0 size mismatch");
  TORCH_CHECK(weight1.sizes() == weight0.sizes(),
              "weight0 and weight1 size mismatch")
  input_size = {B * M, K};
  weight_size = {N, K};
  output_size = {B * M, N};
}

template <typename Gemm>
torch::Tensor cutlass_dual_gemm(
    const torch::Tensor &input, const torch::Tensor &weight0,
    const c10::optional<torch::Tensor> &bias0, const torch::Tensor &weight1,
    const c10::optional<torch::Tensor> &bias1,
    typename Gemm::EpilogueOutputOp2::Params epilogue2_params = {}) {

  using ElementInputA = typename Gemm::ElementA;
  using ElementInputB = typename Gemm::ElementB;
  using ElementOutput = typename Gemm::ElementC;
  using ElementComputeEpilogue =
      typename Gemm::EpilogueOutputOp0::ElementCompute;

  int B, M, K;
  int N;
  cutlass::MatrixCoord input_size, weight_size, output_size;
  get_input_layout(input, weight0, weight1, B, M, K, N, input_size, weight_size,
                   output_size);

  // tensor refs
  cutlass::TensorRef<ElementInputA, LayoutInputA> input_ref(
      reinterpret_cast<ElementInputA *>(input.data_ptr()), LayoutInputA(K));
  cutlass::TensorRef<ElementInputB, LayoutInputB> weight0_ref(
      reinterpret_cast<ElementInputB *>(weight0.data_ptr()), LayoutInputB(K));
  cutlass::TensorRef<ElementInputB, LayoutInputB> weight1_ref(
      reinterpret_cast<ElementInputB *>(weight1.data_ptr()), LayoutInputB(K));
  cutlass::TensorRef<ElementComputeEpilogue, LayoutOutput> bias0_ref(
      reinterpret_cast<ElementComputeEpilogue *>(
          bias0.has_value() ? bias0.value().data_ptr() : nullptr),
      LayoutOutput(N));
  cutlass::TensorRef<ElementComputeEpilogue, LayoutOutput> bias1_ref(
      reinterpret_cast<ElementComputeEpilogue *>(
          bias1.has_value() ? bias1.value().data_ptr() : nullptr),
      LayoutOutput(N));
  bias0_ref.stride(0) = 0;
  bias1_ref.stride(0) = 0;

  torch::Tensor y;
  if (B == 1 && input.dim() != 3) {
    y = torch::empty({M, N},
                     torch::dtype(input.scalar_type()).device(input.device()));
  } else {
    y = torch::empty({B, M, N},
                     torch::dtype(input.scalar_type()).device(input.device()));
  };

  cutlass::TensorRef<ElementOutput, LayoutOutput> output_ref(
      reinterpret_cast<ElementOutput *>(y.data_ptr()), LayoutOutput(N));

  cutlass::TensorRef<ElementOutput, LayoutOutput> nullptr_ref{};

  cutlass::gemm::GemmCoord problem_size(B * M, N, K);

  typename Gemm::Arguments arguments{
      cutlass::gemm::DualGemmMode::kGemm,
      problem_size, // <- problem size of matrix multiplication
      input_ref,
      weight0_ref,
      bias0_ref,
      nullptr_ref,
      weight1_ref,
      bias1_ref,
      nullptr_ref,
      output_ref,
      {ElementComputeEpilogue(1.0),
       ElementComputeEpilogue(
           bias0.has_value() ? 1.0 : 0.0)}, // <- tuple of alpha and beta
      {ElementComputeEpilogue(1.0),
       ElementComputeEpilogue(
           bias0.has_value() ? 1.0 : 0.0)}, // <- tuple of alpha and beta
      epilogue2_params};
  // Allocate workspace memory
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  auto workspace =
      torch::empty({static_cast<int64_t>(workspace_size)},
                   torch::dtype(torch::kUInt8).device(input.device()));

  torch::DeviceGuard device_guard(input.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  cutlass::Status status;
  Gemm gemm_op;

  status = gemm_op.can_implement(arguments);
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "This problem size is not supported by this Gemm implementation: ",
      cutlass::cutlassGetStatusString(status));

  status = gemm_op.initialize(arguments, workspace.data_ptr<uint8_t>());
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "Failed to initialize cutlass gemm: ",
              cutlass::cutlassGetStatusString(status));

  status = gemm_op(stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "Failed to execute cutlass gemm: ",
              cutlass::cutlassGetStatusString(status));
  return y;
}

template <typename at_type> struct cutlass_type { using type = at_type; };

template <> struct cutlass_type<at::Half> { using type = cutlass::half_t; };

template <> struct cutlass_type<at::BFloat16> {
  using type = cutlass::bfloat16_t;
};

template <typename scalar_t> struct acc_type { using type = scalar_t; };

template <> struct acc_type<cutlass::bfloat16_t> { using type = float; };

template <typename at_type, template <typename> class GemmWrapper>
struct CutlassDualGemmLauncher {
  using scalar_t = typename cutlass_type<at_type>::type;
  using acc_t = typename acc_type<scalar_t>::type;
  using config = GemmConfig<scalar_t, acc_t>;
  using Gemm = typename GemmWrapper<config>::Gemm;

  template <typename Func>
  static torch::Tensor
  launch(const torch::Tensor &input, const torch::Tensor &weight0,
         const c10::optional<torch::Tensor> &bias0,
         const torch::Tensor &weight1,
         const c10::optional<torch::Tensor> &bias1, const Func &fallback) {
    auto N = weight0.size(0);
    auto K = weight0.size(1);
    auto M = input.numel() / K;

    if (K % Gemm::kAlignmentA != 0 || K % Gemm::kAlignmentB != 0 ||
        N % Gemm::kAlignmentC != 0) {
      return fallback(input, weight0, bias0, weight1, bias1);
    }
    auto input_ = input.contiguous();
    auto weight0_ = weight0.contiguous();
    auto weight1_ = weight1.contiguous();
    c10::optional<torch::Tensor> bias0_;
    if (bias0.has_value()) {
      bias0_.emplace(bias0.value().contiguous());
    }
    c10::optional<torch::Tensor> bias1_;
    if (bias1.has_value()) {
      bias1_.emplace(bias1.value().contiguous());
    }
    return cutlass_dual_gemm<Gemm>(input_, weight0_, bias0_, weight1_, bias1_);
  }
};

} // namespace

torch::Tensor cutlass_linear_geglu(const torch::Tensor &input,
                                   const torch::Tensor &weight0,
                                   const c10::optional<torch::Tensor> &bias0,
                                   const torch::Tensor &weight1,
                                   const c10::optional<torch::Tensor> &bias1) {
  TORCH_CHECK(input.device().is_cuda(), "input should be on CUDA");
  TORCH_CHECK(input.device() == weight0.device(),
              "input and weight0 should be on the same device");
  TORCH_CHECK(input.scalar_type() == weight0.scalar_type(),
              "input and weight0 should be of the same scalar type");
  TORCH_CHECK(input.device() == weight1.device(),
              "input and weight1 should be on the same device");
  TORCH_CHECK(input.scalar_type() == weight1.scalar_type(),
              "input and weight1 should be of the same scalar type");
  if (bias0.has_value()) {
    TORCH_CHECK(input.device() == bias0.value().device(),
                "input and bias0 should be on the same device");
    TORCH_CHECK(input.scalar_type() == bias0.value().scalar_type(),
                "input and bias0 should have the same scalar type");
  }
  if (bias1.has_value()) {
    TORCH_CHECK(input.device() == bias1.value().device(),
                "input and bias1 should be on the same device");
    TORCH_CHECK(input.scalar_type() == bias1.value().scalar_type(),
                "input and bias1 should have the same scalar type");
  }

  auto fallback = [](const torch::Tensor &input, const torch::Tensor &weight0,
                     const c10::optional<torch::Tensor> &bias0,
                     const torch::Tensor &weight1,
                     const c10::optional<torch::Tensor> &bias1) {
    auto x = cublas_lowp_linear(input, weight0, bias0);
    auto y = cublas_lowp_linear(input, weight1, bias1);
    y = at::gelu_(y);
    return at::mul_out(y, x, y);
  };

  if (input.scalar_type() != at::kHalf &&
      input.scalar_type() != at::kBFloat16) {
    return fallback(input, weight0, bias0, weight1, bias1);
  }

  torch::Tensor output;
  AT_DISPATCH_SWITCH(
      input.scalar_type(), "cutlass_linear_geglu",
      AT_DISPATCH_CASE(
          at::kHalf,
          [&] {
            output =
                CutlassDualGemmLauncher<at::Half, GemmGEGLUWrapper>::launch(
                    input, weight0, bias0, weight1, bias1, fallback);
          });
      AT_DISPATCH_CASE(at::kBFloat16, [&] {
        output =
            CutlassDualGemmLauncher<at::BFloat16, GemmGEGLUWrapper>::launch(
                input, weight0, bias0, weight1, bias1, fallback);
      }));
  return output;
}

torch::Tensor
cutlass_linear_geglu_unified(const torch::Tensor &input,
                             const torch::Tensor &weight,
                             const c10::optional<torch::Tensor> &bias) {
  auto weights = weight.chunk(2, 0);
  c10::optional<torch::Tensor> bias0, bias1;
  if (bias.has_value()) {
    auto biases = bias.value().chunk(2, 0);
    bias0.emplace(biases[0]);
    bias1.emplace(biases[1]);
  }
  return cutlass_linear_geglu(input, weights[0], bias0, weights[1], bias1);
}

} // namespace operators
} // namespace sfast
