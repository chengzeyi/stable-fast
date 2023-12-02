#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_universal.h>
// #include <cutlass/gemm/device/gemm_batched.h>
#include <cutlass/util/reference/host/tensor_compare.h>
#include <cutlass/util/reference/host/tensor_copy.h>
#include <cutlass/util/reference/host/tensor_fill.h>

#include "operators/cublas/cublas_gemm.h"

namespace sfast {
namespace operators {

namespace {

// since int8 only allows RCR, we implement in RCR
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

using MMAOp = cutlass::arch::OpClassTensorOp;
using SimtOp = cutlass::arch::OpClassSimt;

namespace sm80_space {
using SmArch = cutlass::arch::Sm80;
constexpr int NumStages = 4;

template <typename scalar_t, typename acc_t> struct GemmWrapper {
  using ElementA = scalar_t;
  using ElementB = int8_t;
  using ElementOutput = scalar_t;
  using ElementAccumulator = acc_t;
  using ElementComputeEpilogue = scalar_t;

  using Gemm = cutlass::gemm::device::GemmUniversal<
      ElementA, LayoutInputA, ElementB, LayoutInputB, ElementOutput,
      LayoutOutput, ElementAccumulator, MMAOp, SmArch,
      cutlass::gemm::GemmShape<128, 128, 64>,
      cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementComputeEpilogue,
          cutlass::epilogue::thread::ScaleType::NoBetaScaling>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, NumStages,
      128 / cutlass::sizeof_bits<ElementA>::value,
      128 / cutlass::sizeof_bits<ElementB>::value,
      cutlass::arch::OpMultiplyAddMixedInputUpcast,
      cutlass::ComplexTransform::kNone, cutlass::ComplexTransform::kNone>;

  using GemmNoBias = cutlass::gemm::device::GemmUniversal<
      ElementA, LayoutInputA, ElementB, LayoutInputB, ElementOutput,
      LayoutOutput, ElementAccumulator, MMAOp, SmArch,
      cutlass::gemm::GemmShape<128, 128, 64>,
      cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementComputeEpilogue,
          cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, NumStages,
      128 / cutlass::sizeof_bits<ElementA>::value,
      128 / cutlass::sizeof_bits<ElementB>::value,
      cutlass::arch::OpMultiplyAddMixedInputUpcast,
      cutlass::ComplexTransform::kNone, cutlass::ComplexTransform::kNone>;

  using GemmSmall = cutlass::gemm::device::GemmUniversal<
      ElementA, LayoutInputA, ElementB, LayoutInputB, ElementOutput,
      LayoutOutput, ElementAccumulator, MMAOp, SmArch,
      cutlass::gemm::GemmShape<64, 64, 32>,
      cutlass::gemm::GemmShape<32, 32, 32>, cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementComputeEpilogue,
          cutlass::epilogue::thread::ScaleType::NoBetaScaling>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, NumStages,
      128 / cutlass::sizeof_bits<ElementA>::value,
      128 / cutlass::sizeof_bits<ElementB>::value,
      cutlass::arch::OpMultiplyAddMixedInputUpcast,
      cutlass::ComplexTransform::kNone, cutlass::ComplexTransform::kNone>;

  using GemmNoBiasSmall = cutlass::gemm::device::GemmUniversal<
      ElementA, LayoutInputA, ElementB, LayoutInputB, ElementOutput,
      LayoutOutput, ElementAccumulator, MMAOp, SmArch,
      cutlass::gemm::GemmShape<64, 64, 32>,
      cutlass::gemm::GemmShape<32, 32, 32>, cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementComputeEpilogue,
          cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, NumStages,
      128 / cutlass::sizeof_bits<ElementA>::value,
      128 / cutlass::sizeof_bits<ElementB>::value,
      cutlass::arch::OpMultiplyAddMixedInputUpcast,
      cutlass::ComplexTransform::kNone, cutlass::ComplexTransform::kNone>;
};
} // namespace sm80_space

using namespace sm80_space;

void get_input_layout(const torch::Tensor &input, const torch::Tensor &weight,
                      int &B, int &M, int &K, int &N,
                      cutlass::MatrixCoord &input_size,
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
  N = weight.size(0);
  TORCH_CHECK(weight.size(1) == K, "weight size mismatch");
  input_size = {B * M, K};
  weight_size = {N, K};
  output_size = {B * M, N};
}

template <typename Gemm>
torch::Tensor
cutlass_gemm(const torch::Tensor &input, const torch::Tensor &weight,
             const c10::optional<torch::Tensor> &bias, float dq_scale) {

  using ElementInputA = typename Gemm::ElementA;
  using ElementInputB = typename Gemm::ElementB;
  using ElementOutput = typename Gemm::ElementC;
  using ElementComputeEpilogue =
      typename Gemm::EpilogueOutputOp::ElementCompute;

  int B, M, K;
  int N;
  cutlass::MatrixCoord input_size, weight_size, output_size;
  get_input_layout(input, weight, B, M, K, N, input_size, weight_size,
                   output_size);

  // tensor refs
  cutlass::TensorRef<ElementInputA, LayoutInputA> input_ref(
      reinterpret_cast<ElementInputA *>(input.data_ptr()), LayoutInputA(K));
  cutlass::TensorRef<ElementInputB, LayoutInputB> weight_ref(
      reinterpret_cast<ElementInputB *>(weight.data_ptr()), LayoutInputB(K));
  cutlass::TensorRef<ElementComputeEpilogue, LayoutOutput> bias_ref(
      reinterpret_cast<ElementComputeEpilogue *>(
          bias.has_value() ? bias.value().data_ptr() : nullptr),
      LayoutOutput(N));

  input_ref.stride(0) = input.stride(-2);
  weight_ref.stride(0) = weight.stride(0);
  bias_ref.stride(0) = 0;

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

  cutlass::gemm::GemmCoord problem_size(B * M, N, K);

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size, // <- problem size of matrix multiplication
      1,
      {ElementComputeEpilogue(dq_scale),
       ElementComputeEpilogue(
           bias.has_value() ? 1.0 : 0.0)}, // <- tuple of alpha and beta
      input_ref.data(),
      weight_ref.data(),
      bias_ref.data(),
      output_ref.data(),
      0,
      0,
      0,
      0,
      input_ref.stride(0),
      weight_ref.stride(0),
      bias_ref.stride(0),
      output_ref.stride(0)};
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

// ref:
// https://github.com/NVIDIA/cutlass/blob/master/test/unit/gemm/device/gemm_s8t_s8n_s32t_tensor_op_s32_sm80.cu

template <typename at_type> struct cutlass_type { using type = at_type; };

template <> struct cutlass_type<at::Half> { using type = cutlass::half_t; };

template <> struct cutlass_type<at::BFloat16> {
  using type = cutlass::bfloat16_t;
};

template <typename scalar_t> struct acc_type { using type = scalar_t; };

template <> struct acc_type<cutlass::bfloat16_t> { using type = float; };

template <typename at_type> struct CutlassGemmLauncher {
  using scalar_t = typename cutlass_type<at_type>::type;
  using acc_t = typename acc_type<scalar_t>::type;
  using GemmWrapper_ = GemmWrapper<scalar_t, acc_t>;
  using Gemm = typename GemmWrapper_::Gemm;
  using GemmNoBias = typename GemmWrapper_::GemmNoBias;
  using GemmSmall = typename GemmWrapper_::GemmSmall;
  using GemmNoBiasSmall = typename GemmWrapper_::GemmNoBiasSmall;

  static torch::Tensor launch(const torch::Tensor &input,
                              const torch::Tensor &weight,
                              const c10::optional<torch::Tensor> &bias,
                              float dq_scale) {
    auto N = weight.size(0);
    auto K = weight.size(1);
    auto M = input.numel() / K;

    bool use_small_kernel = M <= Gemm::ThreadblockShape::kM ||
                            N <= Gemm::ThreadblockShape::kN ||
                            K <= Gemm::ThreadblockShape::kK;

    if (K % Gemm::kAlignmentA != 0 || K % Gemm::kAlignmentB != 0 ||
        N % Gemm::kAlignmentC != 0) {
      if (K % GemmSmall::kAlignmentA != 0 || K % GemmSmall::kAlignmentB != 0 ||
          N % GemmSmall::kAlignmentC != 0) {
        auto weight_ = input.scalar_type() == at::kFloat
                           ? weight.dequantize()
                           : weight.int_repr()
                                 .to(input.scalar_type())
                                 .mul_(weight.q_scale());
        return cublas_lowp_linear(input, weight_, bias);
      } else {
        use_small_kernel = true;
      }
    }
    auto input_ = input.contiguous();
    auto weight_ = weight.contiguous();
    if (bias.has_value()) {
      c10::optional<torch::Tensor> bias_;
      bias_.emplace(bias.value().contiguous());
      if (use_small_kernel) {
        return cutlass_gemm<GemmSmall>(input_, weight_, bias_, dq_scale);
      } else {
        return cutlass_gemm<Gemm>(input_, weight_, bias_, dq_scale);
      }
    } else {
      if (use_small_kernel) {
        return cutlass_gemm<GemmNoBiasSmall>(input_, weight_, bias, dq_scale);
      } else {
        return cutlass_gemm<GemmNoBias>(input_, weight_, bias, dq_scale);
      }
    }
  }
};

} // namespace

torch::Tensor
cutlass_qlinear_dynamic(const torch::Tensor &input, const torch::Tensor &weight,
                        const c10::optional<torch::Tensor> &bias) {
  TORCH_CHECK(input.device().is_cuda(), "input should be on CUDA");
  TORCH_CHECK(input.device() == weight.device(),
              "input and weight should be on the same device");
  TORCH_CHECK(weight.is_quantized(), "weight should be quantized");
  if (bias.has_value()) {
    TORCH_CHECK(input.device() == bias.value().device(),
                "input and bias should be on the same device");
    TORCH_CHECK(input.scalar_type() == bias.value().scalar_type(),
                "input and bias should have the same scalar type");
  }

  if (input.scalar_type() != at::kHalf &&
      input.scalar_type() != at::kBFloat16) {
    auto weight_ =
        input.scalar_type() == at::kFloat
            ? weight.dequantize()
            : weight.int_repr().to(input.scalar_type()).mul_(weight.q_scale());
    return cublas_lowp_linear(input, weight_, bias);
  }

  torch::Tensor output;
  AT_DISPATCH_SWITCH(
      input.scalar_type(), "cutlass_qlinear_dynamic",
      AT_DISPATCH_CASE(at::kHalf,
                       [&] {
                         output = CutlassGemmLauncher<at::Half>::launch(
                             input, weight, bias, weight.q_scale());
                       });
      AT_DISPATCH_CASE(at::kBFloat16, [&] {
        output = CutlassGemmLauncher<at::BFloat16>::launch(input, weight, bias,
                                                           weight.q_scale());
      }));
  return output;
}

} // namespace operators
} // namespace sfast
