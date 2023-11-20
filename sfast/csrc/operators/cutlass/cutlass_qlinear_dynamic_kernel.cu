#include <torch/extension.h>
#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
// #include "cutlass/gemm/device/gemm_batched.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"

#include "cuda_runtime.h"

#define CUTLASS_CHECK(status)                                                                          \
{                                                                                                  \
    cutlass::Status error = status;                                                                \
    if (error != cutlass::Status::kSuccess)                                                        \
    {                                                                                              \
        std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
        << std::endl;                                                                    \
        exit(EXIT_FAILURE);                                                                        \
    }                                                                                              \
}

#define CUDA_CHECK(status)                                                    \
{                                                                         \
    cudaError_t error = status;                                           \
    if (error != cudaSuccess)                                             \
    {                                                                     \
        std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
        << " at line: " << __LINE__ << std::endl;               \
        exit(EXIT_FAILURE);                                               \
    }                                                                     \
}

#define BLOCKSIZE 1024

namespace sfast
{
namespace operators
{

namespace
{

// half
using Half = cutlass::half_t;
// since int8 only allows RCR, we implement half and fp32 in RCR
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

using MMAOp = cutlass::arch::OpClassTensorOp;
using SimtOp = cutlass::arch::OpClassSimt;

// Number of pipelines you want to use
constexpr int split_k_slices = 1;

namespace sm80_space
{
using SmArch = cutlass::arch::Sm80;
constexpr int NumStages = 3;
namespace half_gemm
{
using ElementInputA = cutlass::half_t;
using ElementInputB = cutlass::half_t;
using ElementOutput = cutlass::half_t;

using ElementAccumulator = cutlass::half_t;
using ElementComputeEpilogue = cutlass::half_t;

using ThreadblockShape = cutlass::gemm::GemmShape<128, 256, 16>; // Threadblock tile shape
                                                                 // This code section describes tile size a warp will compute
using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>;          // Warp tile shape
                                                                 // This code section describes the size of MMA op
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;     // TensorCore instruction shape

using Gemm = cutlass::gemm::device::Gemm<
    ElementInputA, LayoutInputA,
    ElementInputB, LayoutInputB,
    ElementOutput, LayoutOutput,
    ElementAccumulator,
    MMAOp,
    SmArch,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator, ElementComputeEpilogue,
    cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, NumStages>;
};

namespace float_gemm
{
using ElementInputA = float;
using ElementInputB = float;
using ElementOutput = float;

using ElementAccumulator = float;
using ElementComputeEpilogue = float;

using ThreadblockShape = cutlass::gemm::GemmShape<128, 256, 16>; // Threadblock tile shape
                                                                 // This code section describes tile size a warp will compute
using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>; // Warp tile shape
                                                        // This code section describes the size of MMA op
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>; // TensorCore instruction shape

using Gemm = cutlass::gemm::device::Gemm<
    ElementInputA, LayoutInputA,
    ElementInputB, LayoutInputB,
    ElementOutput, LayoutOutput,
    ElementAccumulator,
    MMAOp,
    SmArch,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator, ElementComputeEpilogue,
    cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, NumStages>;
};
}

using namespace sm80_space;

void get_input_layout(const torch::Tensor &input, const torch::Tensor &weight,
                      int &B, int &M, int &K, int &N,
                      cutlass::MatrixCoord &input_size, cutlass::MatrixCoord &weight_size, cutlass::MatrixCoord &output_size)
{
    if (input.dim() == 3)
    {
        B = input.size(0);
        M = input.size(1);
        K = input.size(2);
    }
    else
    {
        B = 1;
        M = input.size(0);
        K = input.size(1);
    }
    // weight is NK
    N = weight.size(0);
    input_size = {B * M, K};
    weight_size = {N, K};
    output_size = {B * M, N};
}

    template <typename Gemm>
torch::Tensor cutlass_gemm(const torch::Tensor &input, const torch::Tensor &weight, const c10::optional<torch::Tensor> &bias, float dq_scale)
{

    using ElementInputA = typename Gemm::ElementA;
    using ElementInputB = typename Gemm::ElementB;
    using ElementOutput = typename Gemm::ElementC;

    int B, M, K;
    int N;
    cutlass::MatrixCoord input_size, weight_size, output_size;
    get_input_layout(input, weight, B, M, K, N, input_size, weight_size, output_size);

    // tensor refs
    cutlass::TensorRef<ElementInputA, LayoutInputA> input_ref(reinterpret_cast<ElementInputA *>(input.data_ptr()), LayoutInputA(K));
    cutlass::TensorRef<ElementInputB, LayoutInputB> weight_ref(reinterpret_cast<ElementInputB *>(weight.data_ptr()), LayoutInputB(K));

    torch::Tensor y;
    if (B == 1 && input.dim() != 3)
    {
        y = torch::empty({M, N}, torch::dtype(input.scalar_type()).device(input.device()));
    }
    else
    {
        y = torch::empty({B, M, N}, torch::dtype(input.scalar_type()).device(input.device()));
    };

    cutlass::TensorRef<ElementOutput, LayoutOutput> output_ref(y.data_ptr<ElementOutput>(), LayoutOutput(N));

    cutlass::gemm::GemmCoord problem_size(B * M, N, K);

    using ElementComputeEpilogue = typename Gemm::EpilogueOutputOp::ElementCompute;

    typename Gemm::Arguments arguments;
    if (bias.has_value()) {
        cutlass::TensorRef<ElementComputeEpilogue, LayoutOutput> bias_ref(reinterpret_cast<ElementComputeEpilogue *>(bias.value().data_ptr()), LayoutOutput(N));
        bias_ref.stride(0) = 0;
        arguments = typename Gemm::Arguments{problem_size,                                                    // <- problem size of matrix multiplication
        input_ref,                                                       // <- reference to matrix A on device
        weight_ref,                                                      // <- reference to matrix B on device
        bias_ref,                                                        // <- reference to matrix C on device
        output_ref,                                                      // <- reference to matrix D on device
        {ElementComputeEpilogue(dq_scale), ElementComputeEpilogue(1.0)}, // <- tuple of alpha and beta
        split_k_slices};                                                 // <- k-dimension split factor
    } else {
        arguments = typename Gemm::Arguments{problem_size,                                                    // <- problem size of matrix multiplication
        input_ref,                                                       // <- reference to matrix A on device
        weight_ref,                                                      // <- reference to matrix B on device
        output_ref,                                                      // <- reference to matrix C on device
        output_ref,                                                      // <- reference to matrix D on device
        {ElementComputeEpilogue(dq_scale), ElementComputeEpilogue(0.0)}, // <- tuple of alpha and beta
        split_k_slices};                                                 // <- k-dimension split factor
    }
    // Allocate workspace memory
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    auto workspace = torch::empty({static_cast<int64_t>(workspace_size)}, torch::dtype(torch::kUInt8).device(input.device()));

    cutlass::Status status;
    Gemm gemm_op;
    status = gemm_op.initialize(arguments, workspace.data_ptr<uint8_t>());
    CUTLASS_CHECK(status);

    status = gemm_op();
    CUTLASS_CHECK(status);
    return y;
}

// ref: https://github.com/NVIDIA/cutlass/blob/master/test/unit/gemm/device/gemm_s8t_s8n_s32t_tensor_op_s32_sm80.cu

torch::Tensor cutlass_gemm_half_interface(const torch::Tensor &input, const torch::Tensor &weight, const c10::optional<torch::Tensor> &bias, float dq_scale)
{
    using Gemm = float_gemm::Gemm;
    return cutlass_gemm<Gemm>(input, weight, bias, dq_scale);
}

torch::Tensor cutlass_gemm_float_interface(const torch::Tensor &input, const torch::Tensor &weight, const c10::optional<torch::Tensor> &bias, float dq_scale)
{
    using Gemm = float_gemm::Gemm;
    return cutlass_gemm<Gemm>(input, weight, bias, dq_scale);
}

}

torch::Tensor cutlass_qlinear_dynamic(const torch::Tensor &input, const torch::Tensor &weight, const c10::optional<torch::Tensor> &bias)
{
    TORCH_CHECK(input.device().is_cuda(), "input should be on CUDA");
    TORCH_CHECK(input.device() == weight.device(), "input and weight should be on the same device");
    TORCH_CHECK(weight.is_quantized(), "weight should be quantized");
    if (bias.has_value())
    {
        TORCH_CHECK(input.device() == bias.value().device(), "input and bias should be on the same device");
        TORCH_CHECK(input.dtype() == bias.value().dtype(), "input and bias should have the same dtype");
    }

    auto input_dtype = input.dtype();
    auto weight_scale = weight.q_scale();

    auto input_ = input.contiguous();
    auto weight_ = weight.contiguous().transpose(-2, -1);
    c10::optional<torch::Tensor> bias_;
    if (bias.has_value())
    {
        bias_.emplace(bias.value().contiguous());
    }

    switch (input.scalar_type())
    {
        case torch::kHalf:
            return cutlass_gemm_half_interface(input_, weight_, bias_, weight_scale);
        case torch::kFloat:
            return cutlass_gemm_float_interface(input_, weight_, bias_, weight_scale);
        default:
            TORCH_CHECK(false, "Unsupported input type");
    }
}

}
}
