#include <torch/extension.h>

#include <torch/csrc/autograd/function_hook.h>
#include <torch/version.h>

#include <ATen/cuda/CUDAConfig.h>
#include <ATen/cudnn/Types.h>
#include <ATen/cudnn/cudnn-wrapper.h>

#include <c10/cuda/CUDACachingAllocator.h>
#if TORCH_VERSION_MAJOR >= 2
#include <c10/core/SymIntArrayRef.h>
#endif

#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <ATen/cuda/EmptyTensor.h>
#include <ATen/cuda/Exceptions.h>   // for CUDNN_CHECK
#include <ATen/cudnn/Descriptors.h> // for TensorDescriptor
#include <ATen/cudnn/Handle.h>      // for getCudnnHandle
#include <ATen/cudnn/Types.h>
#include <ATen/cudnn/Utils.h>
#include <ATen/native/ConvUtils.h>

#include <memory>
#include <mutex>
#include <sstream>
#include <unordered_map>

#include "cudnn_convolution.h"
#include "cudnn_utils.h"

#if CUDNN_VERSION < 8000
#define AT_CUDNN_CONV_BIAS_ADD_FALLBACK
#endif

#if TORCH_VERSION_MAJOR > 1 ||                                                 \
    (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR >= 13)
#define CUDAOutOfMemoryError OutOfMemoryError
#endif

namespace sfast {
namespace operators {

using namespace torch;
using namespace torch::native;
using namespace torch::autograd;

namespace {

constexpr size_t operator"" _TiB(unsigned long long n) {
  return size_t(n) * 1024 * 1024 * 1024 * 1024;
}

struct ConvolutionDescriptor
    : public Descriptor<cudnnConvolutionStruct,
                        &cudnnCreateConvolutionDescriptor,
                        &cudnnDestroyConvolutionDescriptor> {
  void set(cudnnDataType_t dataType, int dim, int *pad, int *stride,
           int *upscale /* aka dilation */, int groups, bool allow_tf32) {
    cudnnDataType_t mathType = dataType;
    if (dataType == CUDNN_DATA_HALF) {
      mathType = CUDNN_DATA_HALF;
    } else if (dataType == CUDNN_DATA_BFLOAT16) {
      mathType = CUDNN_DATA_FLOAT;
    }
    AT_CUDNN_CHECK(
        cudnnSetConvolutionNdDescriptor(mut_desc(), dim, pad, stride, upscale,
                                        CUDNN_CROSS_CORRELATION, mathType));
    AT_CUDNN_CHECK(cudnnSetConvolutionGroupCount(mut_desc(), groups));
    if (dataType == CUDNN_DATA_HALF) {
      AT_CUDNN_CHECK(
          cudnnSetConvolutionMathType(mut_desc(), CUDNN_TENSOR_OP_MATH));
    } else if (dataType == CUDNN_DATA_BFLOAT16) {
      AT_CUDNN_CHECK(
          cudnnSetConvolutionMathType(mut_desc(), CUDNN_TENSOR_OP_MATH));
    } else if (dataType == CUDNN_DATA_FLOAT && !allow_tf32) {
#if defined(CUDNN_VERSION) && CUDNN_VERSION >= 8000
      AT_CUDNN_CHECK(cudnnSetConvolutionMathType(mut_desc(), CUDNN_FMA_MATH));
#else
      AT_CUDNN_CHECK(
          cudnnSetConvolutionMathType(mut_desc(), CUDNN_DEFAULT_MATH));
#endif
    } else {
      AT_CUDNN_CHECK(
          cudnnSetConvolutionMathType(mut_desc(), CUDNN_DEFAULT_MATH));
    }
  }
};

struct ActivationDescriptor
    : public Descriptor<cudnnActivationStruct, &cudnnCreateActivationDescriptor,
                        &cudnnDestroyActivationDescriptor> {
  void set(cudnnActivationMode_t mode) {
    // AT_ASSERT(
    //     mode == CUDNN_ACTIVATION_RELU,
    //     "TODO: support more cuDNN activation modes");
    AT_CUDNN_CHECK(cudnnSetActivationDescriptor(
        mut_desc(), mode, cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN,
        std::numeric_limits<double>::max()));
  }
};

union Constant {
  float f;
  double d;
  Constant(cudnnDataType_t dataType, double value) {
    if (dataType == CUDNN_DATA_HALF || dataType == CUDNN_DATA_BFLOAT16 ||
        dataType == CUDNN_DATA_FLOAT) {
      f = static_cast<float>(value);
    } else {
      d = value;
    }
  }
};

// Hashing machinery for Params
// Fowler–Noll–Vo hash function
// see
// https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
template <typename Params> struct ParamsHash {
  // Params must be a POD because we read out its memory
  // contenst as char* when hashing
  static_assert(std::is_pod<Params>::value, "Params is not POD");

  size_t operator()(const Params &params) const {
    auto ptr = reinterpret_cast<const uint8_t *>(&params);
    uint32_t value = 0x811C9DC5;
    for (const auto i : c10::irange((int)sizeof(Params))) {
      value ^= ptr[i];
      value *= 0x01000193;
    }
    return (size_t)value;
  }
};

template <typename Params> struct ParamsEqual {
  // Params must be a POD because we read out its memory
  // contenst as char* when comparing
  static_assert(std::is_pod<Params>::value, "Params is not POD");

  bool operator()(const Params &a, const Params &b) const {
    auto ptr1 = reinterpret_cast<const uint8_t *>(&a);
    auto ptr2 = reinterpret_cast<const uint8_t *>(&b);
    return memcmp(ptr1, ptr2, sizeof(Params)) == 0;
  }
};

// ---------------------------------------------------------------------
//
// Helper classes
//
// ---------------------------------------------------------------------

// This POD struct is used to let us easily compute hashes of the
// parameters
struct ConvolutionParams {
  c10::DeviceIndex device_id;
  cudnnDataType_t dataType;
  int input_size[2 + max_dim];
  uint8_t input_dim;
  at::MemoryFormat memory_format;
  int weight_size[2 + max_dim];
  int padding[max_dim];
  int stride[max_dim];
  int dilation[max_dim];
  int64_t groups;
  bool deterministic;
  bool allow_tf32;
  // NB: transposed purposely omitted: transposed just swaps
  // forward and backward, so you can reuse the benchmark entry,
};

std::ostream &operator<<(std::ostream &out, const ConvolutionParams &params);

// NB: This can't be a constructor, because then ConvolutionParams
// would not be a POD anymore.
// TODO: Use TensorGeometry here instead of the entire Tensor, which we
// don't actually need.  (OTOH: We can always pass in
// grad_input/grad_output, so this is not very pressing)
void setConvolutionParams(ConvolutionParams *params, const at::Tensor &input,
                          const at::Tensor &weight, IntArrayRef padding,
                          IntArrayRef stride, IntArrayRef dilation,
                          int64_t groups, bool deterministic, bool allow_tf32);

std::string repro_from_args(const ConvolutionParams &args);

// ---------------------------------------------------------------------
//
// ConvolutionParams
//
// ---------------------------------------------------------------------

std::ostream &operator<<(std::ostream &out, const ConvolutionParams &params) {
  out << "ConvolutionParams \n"
      // << "    data_type = " << cudnnTypeToString(params.dataType) << "\n"
      << "    padding = " << ArrayRef<int>{params.padding} << "\n"
      << "    stride = " << ArrayRef<int>{params.stride} << "\n"
      << "    dilation = " << ArrayRef<int>{params.dilation} << "\n"
      << "    groups = " << params.groups << "\n"
      << "    deterministic = " << (params.deterministic ? "true" : "false")
      << "\n"
      << "    allow_tf32 = " << (params.allow_tf32 ? "true" : "false") << "\n";

  return out;
}

// NB: This can't be a constructor, because then ConvolutionParams
// would not be a POD anymore.
// TODO: Use TensorGeometry here instead of the entire Tensor, which we
// don't actually need.  (OTOH: We can always pass in
// grad_input/grad_output, so this is not very pressing)
void setConvolutionParams(ConvolutionParams *params, const at::Tensor &input,
                          const at::Tensor &weight, IntArrayRef padding,
                          IntArrayRef stride, IntArrayRef dilation,
                          int64_t groups, bool deterministic, bool allow_tf32) {

  cudnnDataType_t dataType = getCudnnDataType(input);
  memset(params, 0, sizeof(ConvolutionParams));
  params->device_id = at::cuda::current_device();
  params->dataType = dataType;
  // ASSERT(weight.dim() == input.dim())
  params->input_dim = input.dim();
  params->memory_format = input.suggest_memory_format();
  for (int i = 0; i != params->input_dim; ++i) {
    params->input_size[i] = (int)input.sizes()[i];
    params->weight_size[i] = (int)weight.sizes()[i];
  }
  // ASSERT(padding.size() == stride.size())
  // ASSERT(padding.size() == dilation.size())
  for (size_t i = 0; i != padding.size(); ++i) {
    params->padding[i] = padding[i];
    params->stride[i] = stride[i];
    params->dilation[i] = dilation[i];
  }
  // In principle, we shouldn't parametrize by groups for legacy
  // CuDNN, but it doesn't seem worth the effort to actually do this.
  params->groups = groups;
  params->deterministic = deterministic;
  params->allow_tf32 = allow_tf32;
}

std::string repro_from_args(const ConvolutionParams &params) {
  auto pybool = [](bool b) -> const char * { return b ? "True" : "False"; };
  std::string partial_dtype;
  switch (params.dataType) {
  case CUDNN_DATA_FLOAT:
    partial_dtype = "float";
    break;
  case CUDNN_DATA_DOUBLE:
    partial_dtype = "double";
    break;
  case CUDNN_DATA_HALF:
    partial_dtype = "half";
    break;
  default:
    partial_dtype = "unsupported";
  }
  const std::string full_dtype = "torch." + partial_dtype;
  const int out_channels = params.weight_size[0];
  const int in_channels = params.weight_size[1] * params.groups;
  const size_t dim = params.input_dim;
  const std::string channels_last_xd =
      dim == 4 ? "channels_last" : "channels_last_3d";
  const std::string to_channels_last =
      ((params.memory_format == at::MemoryFormat::ChannelsLast) ||
       (params.memory_format == at::MemoryFormat::ChannelsLast3d))
          ? ".to(memory_format=torch." + channels_last_xd + ")"
          : "";

  std::ostringstream ss;
  ss << "You can try to repro this exception using the following code "
        "snippet. ";
  ss << "If that doesn't trigger the error, please include your original repro "
        "script when reporting this issue.\n\n";
  ss << "import torch\n";
  ss << "torch.backends.cuda.matmul.allow_tf32 = "
     << pybool(at::globalContext().allowTF32CuBLAS()) << "\n";
  ss << "torch.backends.cudnn.benchmark = "
     << pybool(at::globalContext().benchmarkCuDNN()) << "\n";
  ss << "torch.backends.cudnn.deterministic = " << pybool(params.deterministic)
     << "\n";
  ss << "torch.backends.cudnn.allow_tf32 = " << pybool(params.allow_tf32)
     << "\n";
  ss << "data = torch.randn(" << ArrayRef<int>(params.input_size, dim)
     << ", dtype=" << full_dtype << ", ";
  ss << "device='cuda', requires_grad=True)" << to_channels_last << "\n";
  ss << "net = torch.nn.Conv" << dim - 2 << "d(" << in_channels << ", "
     << out_channels << ", ";
  ss << "kernel_size=" << ArrayRef<int>(&params.weight_size[2], dim - 2)
     << ", ";
  ss << "padding=" << ArrayRef<int>(params.padding, dim - 2) << ", ";
  ss << "stride=" << ArrayRef<int>(params.stride, dim - 2) << ", ";
  ss << "dilation=" << ArrayRef<int>(params.dilation, dim - 2) << ", ";
  ss << "groups=" << params.groups << ")\n";
  ss << "net = net.cuda()." << partial_dtype << "()" << to_channels_last
     << "\n";
  ss << "out = net(data)\n";
  ss << "out.backward(torch.randn_like(out))\n";
  ss << "torch.cuda.synchronize()\n\n";

  return ss.str();
}

// Convenience struct for passing around descriptors and data
// pointers
struct ConvolutionArgs {
  cudnnHandle_t handle;
  ConvolutionParams params;
  TensorDescriptor idesc, odesc;
  FilterDescriptor wdesc;
  const Tensor &input, output, weight;
  ConvolutionDescriptor cdesc;

  ConvolutionArgs(const Tensor &input, const Tensor &output,
                  const Tensor &weight)
      : input(input), output(output), weight(weight) {}
};

std::ostream &operator<<(std::ostream &out, const ConvolutionArgs &args) {
  out << repro_from_args(args.params) // already has a trailing newline
      << args.params                  // already has a trailing newline
                     // << "input: " << args.idesc      // already has a
                     // trailing newline
                     // << "output: " << args.odesc     // already has a
                     // trailing newline
                     // << "weight: " << args.wdesc     // already has a
                     // trailing newline
      << "Pointer addresses: "
      << "\n"
      << "    input: " << args.input.data_ptr() << "\n"
      << "    output: " << args.output.data_ptr() << "\n"
      << "    weight: " << args.weight.data_ptr() << "\n";

  return out;
}

// ---------------------------------------------------------------------
//
// Benchmarking
//
// ---------------------------------------------------------------------

// TODO: Use something less heavy duty than a big honking mutex
template <typename T> struct BenchmarkCache {
  std::mutex mutex;
  std::unordered_map<ConvolutionParams, T, ParamsHash<ConvolutionParams>,
                     ParamsEqual<ConvolutionParams>>
      map;

  bool find(const ConvolutionParams &params, T *results) {
    std::lock_guard<std::mutex> guard(mutex);
    auto it = map.find(params);
    if (it == map.end()) {
      return false;
    }
    *results = it->second;
    return true;
  }

  void insert(const ConvolutionParams &params, const T &results) {
    std::lock_guard<std::mutex> guard(mutex);
    map[params] = results;
  }
};

BenchmarkCache<cudnnConvolutionFwdAlgoPerf_t> fwd_algos;
BenchmarkCache<cudnnConvolutionBwdDataAlgoPerf_t> bwd_data_algos;
BenchmarkCache<cudnnConvolutionBwdFilterAlgoPerf_t> bwd_filter_algos;

// TODO: Stop manually allocating CUDA memory; allocate an ATen byte
// tensor instead.
struct Workspace {
  Workspace(size_t size) : size(size), data(NULL) {
    // Sometimes cuDNN returns a workspace size > 2^63, this could makes the
    // allocation of workspace fail with some 64bit indexing error instead of an
    // OOM error. In such case, we manually fail with OOM.
    TORCH_CHECK_WITH(CUDAOutOfMemoryError, size < 1_TiB,
                     "Not enough memory for workspace!");
    data = c10::cuda::CUDACachingAllocator::raw_alloc(size);
  }
  Workspace(const Workspace &) = delete;
  Workspace(Workspace &&) = default;
  Workspace &operator=(Workspace &&) = default;
  ~Workspace() {
    if (data) {
      c10::cuda::CUDACachingAllocator::raw_delete(data);
    }
  }

  size_t size;
  void *data;
};

template <typename perf_t> struct algorithm_search {};

cudnnStatus_t getWorkspaceSize(const ConvolutionArgs &args,
                               cudnnConvolutionFwdAlgo_t algo, size_t *sz) {
  return cudnnGetConvolutionForwardWorkspaceSize(
      args.handle, args.idesc.desc(), args.wdesc.desc(), args.cdesc.desc(),
      args.odesc.desc(), algo, sz);
}
cudnnStatus_t getWorkspaceSize(const ConvolutionArgs &args,
                               cudnnConvolutionBwdDataAlgo_t algo, size_t *sz) {
  return cudnnGetConvolutionBackwardDataWorkspaceSize(
      args.handle, args.wdesc.desc(), args.odesc.desc(), args.cdesc.desc(),
      args.idesc.desc(), algo, sz);
}
cudnnStatus_t getWorkspaceSize(const ConvolutionArgs &args,
                               cudnnConvolutionBwdFilterAlgo_t algo,
                               size_t *sz) {
  return cudnnGetConvolutionBackwardFilterWorkspaceSize(
      args.handle, args.idesc.desc(), args.odesc.desc(), args.cdesc.desc(),
      args.wdesc.desc(), algo, sz);
}

template <typename algo_t>
size_t getMaxWorkspaceSize(const ConvolutionArgs &args, const algo_t *algo,
                           int n_algo) {
  size_t max_ws_size = 0;
  size_t max_block_size = 0;
  const auto device = c10::cuda::current_device();

#if TORCH_VERSION_MAJOR >= 2
  c10::cuda::CUDACachingAllocator::cacheInfo(device, &max_block_size);
#else
  size_t tmp_bytes =
      0; // Only used for filling pointer parameters that aren't used later
  c10::cuda::CUDACachingAllocator::cacheInfo(device, &tmp_bytes,
                                             &max_block_size);
#endif

  for (const auto i : c10::irange(n_algo)) {
    cudnnStatus_t err;
    size_t sz;
    err = getWorkspaceSize(args, algo[i], &sz);
    if (CUDNN_STATUS_SUCCESS != err || sz == 0 || sz < max_ws_size ||
        sz > max_block_size)
      continue;
    max_ws_size = sz;
  }
  return max_ws_size;
}

template <typename perf_t>
std::vector<perf_t> getValidAlgorithms(perf_t *perfResults,
                                       const ConvolutionArgs &args,
                                       int n_algo) {

  // See Note [blocklist fft algorithms for strided dgrad]
#if CUDNN_VERSION < 7500
  bool blocklist = std::is_same<decltype(perfResults[0].algo),
                                cudnnConvolutionBwdDataAlgo_t>::value;
  int stride_dim = args.input.dim() - 2;
  blocklist &= std::any_of(std::begin(args.params.stride),
                           std::begin(args.params.stride) + stride_dim,
                           [=](int n) { return n != 1; });
#endif

  std::vector<perf_t> result;
  result.reserve(n_algo);
  for (const auto i : c10::irange(n_algo)) {
    perf_t perf = perfResults[i];

    // TODO: Shouldn't all returned results be successful?
    // Double check documentation for cudnnFindConvolutionForwardAlgorithmEx
    if (perf.status == CUDNN_STATUS_SUCCESS) {
      if (!args.params.deterministic ||
          perf.determinism == CUDNN_DETERMINISTIC) {

        // See Note [blocklist fft algorithms for strided dgrad]
#if CUDNN_VERSION < 7500
        bool skip = blocklist;
        skip &=
            (static_cast<cudnnConvolutionBwdDataAlgo_t>(perfResults[i].algo) ==
                 CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING ||
             static_cast<cudnnConvolutionBwdDataAlgo_t>(perfResults[i].algo) ==
                 CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT);
        if (skip) {
          continue;
        }
#endif

        result.push_back(perf);
      }
    }
  }
  TORCH_CHECK(result.size() > 0,
              "no valid convolution algorithms available in CuDNN");
  return result;
}

template <> struct algorithm_search<cudnnConvolutionFwdAlgoPerf_t> {
  using perf_t = cudnnConvolutionFwdAlgoPerf_t;
  using algo_t = cudnnConvolutionFwdAlgo_t;

  static constexpr auto DEFAULT_ALGO =
      CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  static BenchmarkCache<perf_t> &cache() { return fwd_algos; }

  static std::vector<perf_t> findAlgorithms(const ConvolutionArgs &args,
                                            bool benchmark) {
    static const algo_t algos[] = {
        CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
        CUDNN_CONVOLUTION_FWD_ALGO_FFT,
        CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
        CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
        CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
        CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
    };
    static constexpr int num_algos = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
    static_assert(sizeof(algos) / sizeof(algos[0]) == num_algos,
                  "Missing cuDNN convolution forward algorithms");
    int perf_count;
    std::unique_ptr<perf_t[]> perf_results(new perf_t[num_algos]);
    if (!benchmark) {
      AT_CUDNN_CHECK_WITH_SHAPES(cudnnGetConvolutionForwardAlgorithm_v7(
                                     args.handle, args.idesc.desc(),
                                     args.wdesc.desc(), args.cdesc.desc(),
                                     args.odesc.desc(), num_algos, &perf_count,
                                     perf_results.get()),
                                 args);
    } else {
      size_t max_ws_size = getMaxWorkspaceSize(args, algos, num_algos);
      Workspace ws(max_ws_size);
      at::cuda::errorIfCapturingCudnnBenchmark("cudnnFind");
      AT_CUDNN_CHECK_WITH_SHAPES(
          cudnnFindConvolutionForwardAlgorithmEx(
              args.handle, args.idesc.desc(), args.input.data_ptr(),
              args.wdesc.desc(), args.weight.data_ptr(), args.cdesc.desc(),
              args.odesc.desc(), args.output.data_ptr(), num_algos, &perf_count,
              perf_results.get(), ws.data, ws.size),
          args);

      // Free the cached blocks in our caching allocator. They are
      // needed here because the above benchmarking uses a huge amount of
      // memory, e.g. a few GBs.
      c10::cuda::CUDACachingAllocator::emptyCache();
    }
    return getValidAlgorithms<perf_t>(perf_results.get(), args, perf_count);
  }

  static void getWorkspaceSize(const ConvolutionArgs &args, algo_t algo,
                               size_t *workspaceSize) {
    AT_CUDNN_CHECK_WITH_SHAPES(cudnnGetConvolutionForwardWorkspaceSize(
                                   args.handle, args.idesc.desc(),
                                   args.wdesc.desc(), args.cdesc.desc(),
                                   args.odesc.desc(), algo, workspaceSize),
                               args);
  }
};

template <> struct algorithm_search<cudnnConvolutionBwdDataAlgoPerf_t> {
  using perf_t = cudnnConvolutionBwdDataAlgoPerf_t;
  using algo_t = cudnnConvolutionBwdDataAlgo_t;

  static constexpr auto DEFAULT_ALGO = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
  static BenchmarkCache<perf_t> &cache() { return bwd_data_algos; }

  static std::vector<perf_t> findAlgorithms(const ConvolutionArgs &args,
                                            bool benchmark) {
    static const algo_t algos[] = {
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED};
    static constexpr int num_algos = CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT;
    static_assert(sizeof(algos) / sizeof(algos[0]) == num_algos,
                  "Missing cuDNN convolution backward data algorithms.");
    int perf_count;
    std::unique_ptr<perf_t[]> perf_results(new perf_t[num_algos]);
    if (!benchmark) {
      AT_CUDNN_CHECK_WITH_SHAPES(cudnnGetConvolutionBackwardDataAlgorithm_v7(
                                     args.handle, args.wdesc.desc(),
                                     args.odesc.desc(), args.cdesc.desc(),
                                     args.idesc.desc(), num_algos, &perf_count,
                                     perf_results.get()),
                                 args);
    } else {
      size_t max_ws_size = getMaxWorkspaceSize(args, algos, num_algos);
      Workspace ws(max_ws_size);
      at::cuda::errorIfCapturingCudnnBenchmark("cudnnFind");
      AT_CUDNN_CHECK_WITH_SHAPES(
          cudnnFindConvolutionBackwardDataAlgorithmEx(
              args.handle, args.wdesc.desc(), args.weight.data_ptr(),
              args.odesc.desc(), args.output.data_ptr(), args.cdesc.desc(),
              args.idesc.desc(), args.input.data_ptr(), num_algos, &perf_count,
              perf_results.get(), ws.data, ws.size),
          args);

      // Free the cached blocks in our caching allocator. They are
      // needed here because the above benchmarking uses a huge amount of
      // memory, e.g. a few GBs.
      c10::cuda::CUDACachingAllocator::emptyCache();
    }
    return getValidAlgorithms<perf_t>(perf_results.get(), args, perf_count);
  }

  static void getWorkspaceSize(const ConvolutionArgs &args,
                               cudnnConvolutionBwdDataAlgo_t algo,
                               size_t *workspaceSize) {
    AT_CUDNN_CHECK_WITH_SHAPES(cudnnGetConvolutionBackwardDataWorkspaceSize(
                                   args.handle, args.wdesc.desc(),
                                   args.odesc.desc(), args.cdesc.desc(),
                                   args.idesc.desc(), algo, workspaceSize),
                               args);
  }
};

template <> struct algorithm_search<cudnnConvolutionBwdFilterAlgoPerf_t> {
  using perf_t = cudnnConvolutionBwdFilterAlgoPerf_t;
  using algo_t = cudnnConvolutionBwdFilterAlgo_t;

  static constexpr auto DEFAULT_ALGO = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;

  static BenchmarkCache<perf_t> &cache() { return bwd_filter_algos; }

  static std::vector<perf_t> findAlgorithms(const ConvolutionArgs &args,
                                            bool benchmark) {
    static const algo_t algos[] = {
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING,
    };
    // NOTE: - 1 because ALGO_WINOGRAD is not implemented
    static constexpr int num_algos =
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT - 1;
    static_assert(sizeof(algos) / sizeof(algos[0]) == num_algos,
                  "Missing cuDNN convolution backward filter algorithms.");
    std::unique_ptr<perf_t[]> perf_results(new perf_t[num_algos]);
    int perf_count;
    if (!benchmark) {
      AT_CUDNN_CHECK_WITH_SHAPES(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
                                     args.handle, args.idesc.desc(),
                                     args.odesc.desc(), args.cdesc.desc(),
                                     args.wdesc.desc(), num_algos, &perf_count,
                                     perf_results.get()),
                                 args);
    } else {
      size_t max_ws_size = getMaxWorkspaceSize(args, algos, num_algos);
      Workspace ws(max_ws_size);
      at::cuda::errorIfCapturingCudnnBenchmark("cudnnFind");
      AT_CUDNN_CHECK_WITH_SHAPES(
          cudnnFindConvolutionBackwardFilterAlgorithmEx(
              args.handle, args.idesc.desc(), args.input.data_ptr(),
              args.odesc.desc(), args.output.data_ptr(), args.cdesc.desc(),
              args.wdesc.desc(), args.weight.data_ptr(), num_algos, &perf_count,
              perf_results.get(), ws.data, ws.size),
          args);

      // Free the cached blocks in our caching allocator. They are
      // needed here because the above benchmarking uses a huge amount of
      // memory, e.g. a few GBs.
      c10::cuda::CUDACachingAllocator::emptyCache();
    }
    return getValidAlgorithms<perf_t>(perf_results.get(), args, perf_count);
  }

  static void getWorkspaceSize(const ConvolutionArgs &args, algo_t algo,
                               size_t *workspaceSize) {
    AT_CUDNN_CHECK_WITH_SHAPES(cudnnGetConvolutionBackwardFilterWorkspaceSize(
                                   args.handle, args.idesc.desc(),
                                   args.odesc.desc(), args.cdesc.desc(),
                                   args.wdesc.desc(), algo, workspaceSize),
                               args);
  }
};

template <typename perf_t> class AlgoIterator {
  using search = algorithm_search<perf_t>;
  const ConvolutionArgs &args;
  bool benchmark;

public:
  AlgoIterator(const ConvolutionArgs &args, bool benchmark)
      : args(args), benchmark(benchmark) {}

  static std::vector<perf_t> onlyDefaultAlgorithm(const ConvolutionArgs &args) {
    std::vector<perf_t> perfResults(1);
    perfResults[0].algo = search::DEFAULT_ALGO;
    if (args.params.dataType == CUDNN_DATA_HALF ||
        args.params.dataType == CUDNN_DATA_BFLOAT16) {
      perfResults[0].mathType = CUDNN_TENSOR_OP_MATH;
    } else {
      perfResults[0].mathType = CUDNN_DEFAULT_MATH;
#if defined(CUDNN_VERSION) && CUDNN_VERSION >= 8000
      if (args.params.dataType == CUDNN_DATA_FLOAT && !args.params.allow_tf32) {
        perfResults[0].mathType = CUDNN_FMA_MATH;
      }
#endif
    }
    search::getWorkspaceSize(args, perfResults[0].algo,
                             &(perfResults[0].memory));
    return perfResults;
  }

  void try_all(std::function<void(const perf_t &perf)> f) {
    bool only_use_default = args.params.deterministic && !benchmark;

    auto &cache = search::cache();
    perf_t algoPerf;
    if (!only_use_default && cache.find(args.params, &algoPerf)) {
      try {
        f(algoPerf);
        return;
      } catch (c10::CUDAOutOfMemoryError &e) {
        cudaGetLastError(); // clear CUDA error
      }
    }

    auto perfResults = only_use_default
                           ? onlyDefaultAlgorithm(args)
                           : search::findAlgorithms(args, benchmark);
    for (auto &algoPerf : perfResults) {
      try {
        f(algoPerf);
        cache.insert(args.params, algoPerf);
        return;
      } catch (c10::CUDAOutOfMemoryError &e) {
        cudaGetLastError(); // clear CUDA error
      } catch (c10::CuDNNError &e) {
        cudaGetLastError(); // clear CUDA error
      }
    }
    TORCH_CHECK(false,
                "Unable to find a valid cuDNN algorithm to run convolution");
  }
};

inline Tensor allocate_workspace(size_t size, const Tensor &other) {
  // Sometimes cuDNN returns a workspace size > 2^63, this could makes the
  // allocation of workspace fail with some 64bit indexing error instead of an
  // OOM error. In such case, we manually fail with OOM.
  TORCH_CHECK_WITH(CUDAOutOfMemoryError, size < 1_TiB,
                   "Not enough memory for workspace!");
  return at::empty({static_cast<int64_t>(size)}, other.options().dtype(kByte));
}

// NOTE [ raw_cudnn_convolution_forward_out ]
//
//    - raw_cudnn_convolution_forward_out (Tensor)
//      Functiont that handles tensors that are too large to use 32bit indexing.
//      It just split the tensor and dispatches to
//      `raw_cudnn_convolution_forward_out_32bit`.
//
//    - raw_cudnn_convolution_forward_out_32bit (Tensor)
//      Low level function which invokes CuDNN, and takes an output
//      tensor which is directly written to (thus _out).
//

// ---------------------------------------------------------------------
//
// Splitting to 32bit
//
// ---------------------------------------------------------------------

template <typename func_t>
static inline void split_batch_dim_to_32bit_out(
    const at::Tensor &output, const at::Tensor &input, const at::Tensor &weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, bool benchmark, bool deterministic, bool allow_tf32,
    int64_t max_worksize, func_t func_32bit) {
  constexpr int64_t int_max = std::numeric_limits<int>::max();
  const int64_t ni = input.numel();
  const int64_t no = output.numel();
  // Assume the shape of the tensor is (N, C, D1, D2, ...)
  // if N * C * D1 * D2 * ... <= int_max, then no need to split at all
  if (ni <= int_max && no <= int_max) {
    func_32bit(output, input, weight, padding, stride, dilation, groups,
               benchmark, deterministic, allow_tf32);
    return;
  }
  // else, if C * D1 * D2 * ... <= int_max, then we just need to split across
  // the N dimension
  //
  // Here we use a simple heuristics to determine the size of each split
  // We don't max out the 2^31 address space because this number is super
  // large and very likely to get an OOM.
  int64_t n = output.size(0);
  int64_t max_inner_size = std::max<int64_t>(ni, no) / n;
  int64_t split_size = std::max<int64_t>(max_worksize / max_inner_size, 1L);
  int64_t num_splits = (n + split_size - 1) / split_size;
  if (split_size * max_inner_size < int_max) {
    for (const auto i : c10::irange(num_splits)) {
      int64_t start = split_size * i;
      int64_t split_size_ = std::min<int64_t>(split_size, n - start);
      Tensor input_ = input.narrow(0, start, split_size_);
      Tensor output_ = output.narrow(0, start, split_size_);
      func_32bit(output_, input_, weight, padding, stride, dilation, groups,
                 benchmark, deterministic, allow_tf32);
    }
    return;
  }
  // If control flow reaches here, this means even splitting N is not enough,
  // then things starts to become complicated: For example, for conv2d, there
  // following questions needs to be considered.
  // - Is the memory layout NCHW or NHWC ?
  // - If the conv is NCHW -> NC'H'W', then should we
  //   - split only NC?
  //   - split only N'C'?
  //   - split both?
  // - If the conv is NHWC, then we need to split across H, we need to be very
  // careful about the boundary condition
  //   to make sure that the boundary is handled correctly.
  // - If we decide to make these splits, is the memory contiguous? Do we need
  // to copy the memory? Considering the complexity of this issue, it is better
  // not to use cuDNN for this case
  TORCH_INTERNAL_ASSERT(false, "This case should not be dispatched to cuDNN.");
}

#if defined(CUDNN_VERSION) && CUDNN_VERSION >= 8000
#define ASSERT_CORRECT_PRECISION(math_type)                                    \
  if (args.params.dataType == CUDNN_DATA_FLOAT) {                              \
    TORCH_INTERNAL_ASSERT(args.params.allow_tf32 ||                            \
                          math_type == CUDNN_FMA_MATH);                        \
  }
#else
#define ASSERT_CORRECT_PRECISION(math_type)
#endif // CUDNN_VERSION >= 8000

void raw_cudnn_convolution_forward_out_32bit(
    const Tensor &output, const Tensor &input, const Tensor &weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, bool benchmark, bool deterministic, bool allow_tf32) {
  DeviceGuard device_guard(input.device());

  auto dataType = getCudnnDataType(input);

  ConvolutionArgs args{input, output, weight};
  args.handle = getCudnnHandle();
  setConvolutionParams(&args.params, input, weight, padding, stride, dilation,
                       groups, deterministic, allow_tf32);
  at::MemoryFormat memory_format =
      cudnn_conv_suggest_memory_format(input, weight);
  args.idesc.set(input, memory_format);
  args.wdesc.set(weight, memory_format, 0);
  args.odesc.set(output, memory_format);
  args.cdesc.set(dataType, input.dim() - 2, args.params.padding,
                 args.params.stride, args.params.dilation, args.params.groups,
                 args.params.allow_tf32);

  // TODO: when we do legacy group convolution support, we'll repeatedly
  // reinitialize the workspace for each convolution we do.  This is
  // wasteful; we'd rather reuse the workspace.  OTOH, legacy group
  // convolution support is already pretty slow, so this might not
  // matter.  (This applies to raw_cudnn_convolution_backward_input as well.)
  AlgoIterator<cudnnConvolutionFwdAlgoPerf_t>(args, benchmark)
      .try_all([&](const cudnnConvolutionFwdAlgoPerf_t &fwdAlgPerf) {
        Tensor workspace = allocate_workspace(fwdAlgPerf.memory, input);

        // update convDesc mathType since cudnn 7.4+ now requires both algo
        // + mathType to figure out whether to use Tensor core kernels or
        // not See Note [behavior of cudnnFind and cudnnGet]
        ASSERT_CORRECT_PRECISION(fwdAlgPerf.mathType);
        AT_CUDNN_CHECK_WITH_SHAPES(
            cudnnSetConvolutionMathType(args.cdesc.mut_desc(),
                                        fwdAlgPerf.mathType),
            args);

        Constant one(dataType, 1);
        Constant zero(dataType, 0);

        AT_CUDNN_CHECK_WITH_SHAPES(
            cudnnConvolutionForward(
                args.handle, &one, args.idesc.desc(), input.data_ptr(),
                args.wdesc.desc(), weight.data_ptr(), args.cdesc.desc(),
                fwdAlgPerf.algo, workspace.data_ptr(), fwdAlgPerf.memory, &zero,
                args.odesc.desc(), output.data_ptr()),
            args, "Forward algorithm: ", static_cast<int>(fwdAlgPerf.algo),
            "\n");
      });
}

void raw_cudnn_convolution_forward_out(
    const Tensor &output, const Tensor &input, const Tensor &weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, bool benchmark, bool deterministic, bool allow_tf32) {
  split_batch_dim_to_32bit_out(output, input, weight, padding, stride, dilation,
                               groups, benchmark, deterministic, allow_tf32,
                               1024 * 1024 * 256,
                               raw_cudnn_convolution_forward_out_32bit);
}

void raw_cudnn_convolution_add_out(
    const Tensor &output, const Tensor &input, const Tensor &weight,
    const Tensor &z, float alpha, const Tensor &bias, IntArrayRef stride,
    IntArrayRef padding, IntArrayRef dilation, int64_t groups, bool benchmark,
    bool deterministic, bool allow_tf32,
    cudnnActivationMode_t activation_mode = CUDNN_ACTIVATION_IDENTITY,
    bool with_bias = true) {
  DeviceGuard device_guard(input.device());

  auto dataType = getCudnnDataType(input);

  ConvolutionArgs args{input, output, weight};
  args.handle = getCudnnHandle();
  setConvolutionParams(&args.params, input, weight, padding, stride, dilation,
                       groups, deterministic, allow_tf32);
  at::MemoryFormat memory_format =
      cudnn_conv_suggest_memory_format(input, weight);
  args.idesc.set(input, memory_format);
  args.wdesc.set(weight, memory_format, 0);
  args.odesc.set(output, memory_format);
  args.cdesc.set(dataType, input.dim() - 2, args.params.padding,
                 args.params.stride, args.params.dilation, args.params.groups,
                 args.params.allow_tf32);

  TensorDescriptor zdesc;
  zdesc.set(z, memory_format);

  TensorDescriptor bdesc;
  bdesc.set(bias.expand({1, bias.size(0)}), memory_format, output.dim());

  ActivationDescriptor adesc;
  adesc.set(activation_mode);

  AlgoIterator<cudnnConvolutionFwdAlgoPerf_t>(args, benchmark)
      .try_all([&](const cudnnConvolutionFwdAlgoPerf_t &fwdAlgPerf) {
        Tensor workspace = allocate_workspace(fwdAlgPerf.memory, input);

        // update convDesc mathType since cudnn 7.4+ now requires both algo
        // + mathType to figure out whether to use Tensor core kernels or
        // not See Note [behavior of cudnnFind and cudnnGet]
        ASSERT_CORRECT_PRECISION(fwdAlgPerf.mathType);
        AT_CUDNN_CHECK_WITH_SHAPES(
            cudnnSetConvolutionMathType(args.cdesc.mut_desc(),
                                        fwdAlgPerf.mathType),
            args);

        Constant one(dataType, 1);
        Constant alpha_(dataType, alpha);

        // If mathType is not using Tensor Core, then the fusion is extremely
        // slow
        if ((fwdAlgPerf.mathType == CUDNN_TENSOR_OP_MATH ||
             fwdAlgPerf.mathType == CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION) &&
            (activation_mode != CUDNN_ACTIVATION_IDENTITY ||
             fwdAlgPerf.algo ==
                 CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM)) {
          AT_CUDNN_CHECK_WITH_SHAPES(
              cudnnConvolutionBiasActivationForward(
                  args.handle, &one, args.idesc.desc(), input.data_ptr(),
                  args.wdesc.desc(), weight.data_ptr(), args.cdesc.desc(),
                  fwdAlgPerf.algo, workspace.data_ptr(), fwdAlgPerf.memory,
                  &alpha_, zdesc.desc(), z.data_ptr(), bdesc.desc(),
                  bias.data_ptr(), adesc.desc(), args.odesc.desc(),
                  output.data_ptr()),
              args, // "zdesc: ", zdesc, "bdesc: ", bdesc,
              "cudnnConvolutionBiasActivationForward: ",
              static_cast<int>(fwdAlgPerf.algo), "\n");
        } else {
          Constant zero(dataType, 0);
          AT_CUDNN_CHECK_WITH_SHAPES(
              cudnnConvolutionForward(
                  args.handle, &one, args.idesc.desc(), input.data_ptr(),
                  args.wdesc.desc(), weight.data_ptr(), args.cdesc.desc(),
                  fwdAlgPerf.algo, workspace.data_ptr(), fwdAlgPerf.memory,
                  &zero, args.odesc.desc(), output.data_ptr()),
              args, // "bdesc: ", bdesc,
              "cudnnConvolutionForward: ", static_cast<int>(fwdAlgPerf.algo),
              "\n");

          if (with_bias) {
            AT_CUDNN_CHECK(
                cudnnAddTensor(args.handle, &one, bdesc.desc(), bias.data_ptr(),
                               &one, args.odesc.desc(), output.data_ptr()));
          }
          if (alpha != 0.0f) {
            AT_CUDNN_CHECK(cudnnAddTensor(args.handle, &alpha_, zdesc.desc(),
                                          z.data_ptr(), &one, args.odesc.desc(),
                                          output.data_ptr()));
          }
          if (activation_mode != CUDNN_ACTIVATION_IDENTITY) {
            AT_CUDNN_CHECK(cudnnActivationForward(
                args.handle, adesc.desc(), &one, args.odesc.desc(),
                output.data_ptr(), &zero, args.odesc.desc(),
                output.data_ptr()));
          }
        }
      });
}

void raw_cudnn_convolution_add_fallback_out(
    const Tensor &output, const Tensor &input, const Tensor &weight,
    const Tensor &z, float alpha, const Tensor &bias, IntArrayRef stride,
    IntArrayRef padding, IntArrayRef dilation, int64_t groups, bool benchmark,
    bool deterministic, bool allow_tf32) {

  // cuDNN Conv-Bias-Activation:
  // y = act ( alpha1 * conv(x) + alpha2 * z + bias )
  // In pytorch function `raw_cudnn_convolution_add_relu_out`: alpha1 is 1,
  // alpha 2 is `float alpha`

  raw_cudnn_convolution_forward_out(output, input, weight, padding, stride,
                                    dilation, groups, benchmark, deterministic,
                                    allow_tf32);
  at::Tensor alpha_mul_z_add_bias = at::native::reshape_bias(input.dim(), bias);
  if (alpha != 0.0) {
    alpha_mul_z_add_bias = alpha_mul_z_add_bias.add(z, alpha);
  }
  output.add_(alpha_mul_z_add_bias);
}

} // namespace

Tensor cudnn_convolution_bias_add_activation(
    const Tensor &input_t, const Tensor &weight_t,
    const c10::optional<Tensor> &z_t, const c10::optional<Scalar> &alpha,
    const c10::optional<Tensor> &bias_t, IntArrayRef stride,
    IntArrayRef padding, IntArrayRef dilation, int64_t groups,
    cudnnActivationMode_t activation_mode = CUDNN_ACTIVATION_IDENTITY) {
  CheckedFrom c = "cudnn_convolution_bias_add_activation";
  TensorArg input_arg{input_t, "input", 1}, weight_arg{weight_t, "weight", 2};
  checkAllSameGPU(c, {input_arg, weight_arg});
  checkAllSameType(c, {input_arg, weight_arg});

  auto memory_format = cudnn_conv_suggest_memory_format(input_t, weight_t);
  const Tensor input = input_t.contiguous(memory_format);
  const Tensor weight = weight_t.contiguous(memory_format);

  // FuseFrozenConvAddRelu performs some tensor shape checking
  Tensor output_t =
      at::detail::empty_cuda(conv_output_size(input.sizes(), weight.sizes(),
                                              padding, stride, dilation),
                             input.options().memory_format(memory_format));
  if (output_t.numel() == 0) {
    return output_t;
  }

  Tensor z;
  if (z_t.has_value()) {
    z = z_t.value();

    TensorArg z_arg{z, "z", 3};
    checkAllSameGPU(c, {input_arg, z_arg});
    checkAllSameType(c, {input_arg, z_arg});
    // TensorArg output_arg{output_t, "output", 0};
    // checkSameSize(c, output_arg, z_arg);

    if (z.sizes() != output_t.sizes()) {
      // Have to broadcast or an invalid memory access will occur
      z = z.broadcast_to(output_t.sizes());
    }
    if (z.suggest_memory_format() != memory_format) {
      z = z.to(memory_format);
    }
    z = z.contiguous(memory_format);
  } else {
    z = output_t;
  }

  auto &ctx = at::globalContext();
  bool allow_tf32 = ctx.allowTF32CuDNN();
  bool benchmark = ctx.benchmarkCuDNN();
  float _alpha = alpha.has_value() ? alpha.value().to<float>() : 1.0f;
  if (!z_t.has_value()) {
    _alpha = 0.0f;
  }
  Tensor _bias;
  if (bias_t.has_value()) {
    _bias = bias_t.value();

    TensorArg bias_arg{_bias, "bias", 4};
    checkAllSameGPU(c, {input_arg, bias_arg});
    checkAllSameType(c, {input_arg, bias_arg});
    TORCH_CHECK(_bias.dim() == 1 && _bias.size(0) == weight.size(0),
                "bias should have shape [", weight.size(0), "]");
  } else {
    _bias =
#if TORCH_VERSION_MAJOR > 1 ||                                                 \
    (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR >= 13)
        at::zeros
#else
        at::native::zeros
#endif
        ({output_t.size(1)},
         optTypeMetaToScalarType(output_t.options().dtype_opt()),
         output_t.options().layout_opt(), output_t.options().device_opt(),
         output_t.options().pinned_memory_opt());
  }

#ifdef AT_CUDNN_CONV_BIAS_ADD_FALLBACK
  raw_cudnn_convolution_add_fallback_out(output_t, input, weight, z, _alpha,
                                         _bias, stride, padding, dilation,
                                         groups, benchmark,
                                         false,     // deterministic
                                         allow_tf32 // allow_tf32
  );
  switch (activation_mode) {
  case CUDNN_ACTIVATION_IDENTITY:
    break;
  case CUDNN_ACTIVATION_SIGMOID:
    output_t.sigmoid_();
    break;
  case CUDNN_ACTIVATION_RELU:
    output_t.relu_();
    break;
  case CUDNN_ACTIVATION_TANH:
    output_t.tanh_();
    break;
  default:
    TORCH_CHECK(false, "Unsupported activation mode: ", activation_mode);
  }
#else  // AT_CUDNN_CONV_BIAS_ADD_FALLBACK
  raw_cudnn_convolution_add_out(output_t, input, weight, z, _alpha, _bias,
                                stride, padding, dilation, groups, benchmark,
                                false,      // deterministic
                                allow_tf32, // allow_tf32
                                activation_mode, bias_t.has_value());
#endif // AT_CUDNN_CONV_BIAS_ADD_FALLBACK

  return output_t;
}

inline std::vector<int64_t> expand_param_if_needed(IntArrayRef list_param,
                                                   const char *param_name,
                                                   int64_t expected_dim) {
  if (list_param.size() == 1) {
    return std::vector<int64_t>(expected_dim, list_param[0]);
  } else if ((int64_t)list_param.size() != expected_dim) {
    std::ostringstream ss;
    ss << "expected " << param_name << " to be a single integer value or a "
       << "list of " << expected_dim << " values to match the convolution "
       << "dimensions, but got " << param_name << "=" << list_param;
    AT_ERROR(ss.str());
  } else {
    return list_param.vec();
  }
}

inline auto view4d(const at::Tensor &tensor) -> at::Tensor {
  TORCH_CHECK(tensor.ndimension() == 3, "expected 3D tensor, got tensor with ",
              tensor.ndimension(), " dimensions instead");
  return tensor.unsqueeze(2);
}

inline auto view3d(const at::Tensor &tensor) -> at::Tensor {
  TORCH_CHECK(tensor.ndimension() == 4, "expected 4D tensor, got tensor with ",
              tensor.ndimension(), " dimensions instead");
  return tensor.squeeze(2);
}

#if TORCH_VERSION_MAJOR >= 2
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
// This struct is templated so that we can run backend selection in a dynamic
// shapes context; all of the real kernel selection in eager mode runs with
// int64_t
template <typename T> struct _ConvParams {
  std::vector<int64_t> stride;
  std::vector<T> padding;
  std::vector<int64_t> dilation;
  bool transposed;
  std::vector<T> output_padding;
  int groups;
  bool benchmark;
  bool deterministic;
  bool cudnn_enabled;
  bool allow_tf32;
};

using ConvParams = _ConvParams<int64_t>;
#endif

static auto view1d_as_2d(ConvParams &params) -> void {
  if (params.stride.size() == 1) {
    params.stride.insert(params.stride.begin(), 1);
    params.padding.insert(params.padding.begin(), 0);
    params.dilation.insert(params.dilation.begin(), 1);
    params.output_padding.insert(params.output_padding.begin(), 0);
  }
}

#if TORCH_VERSION_MAJOR >= 2
static ConvBackend
select_conv_backend(const Tensor &input_r, const Tensor &weight_r,
                    const c10::optional<Tensor> &bias_opt, IntArrayRef stride_,
                    IntArrayRef padding_, IntArrayRef dilation_,
                    bool transposed_, IntArrayRef output_padding_,
                    int64_t groups_) {
  return torch::native::select_conv_backend(
      input_r, weight_r, bias_opt,
#if TORCH_VERSION_MINOR >= 2
      fromIntArrayRefUnchecked
#endif
      (stride_),
      fromIntArrayRefUnchecked(padding_),
#if TORCH_VERSION_MINOR >= 2
      fromIntArrayRefUnchecked
#endif
      (dilation_),
      transposed_, fromIntArrayRefUnchecked(output_padding_), groups_,
      c10::nullopt);
}
#endif

Tensor cudnn_convolution_bias_add_activation_with_fallback_forward(
    const Tensor &input_t, const Tensor &weight_t,
    const c10::optional<Tensor> &z_t, const c10::optional<Scalar> &alpha,
    const c10::optional<Tensor> &bias_t, IntArrayRef stride,
    IntArrayRef padding, IntArrayRef dilation, bool transposed,
    IntArrayRef output_padding, int64_t groups,
    cudnnActivationMode_t activation_mode = CUDNN_ACTIVATION_IDENTITY) {
  // bool need_backward = GradMode::is_enabled() &&
  //                      (input_t.requires_grad() || weight_t.requires_grad()
  //                      ||
  //                       (bias_t.has_value() &&
  //                       bias_t.value().requires_grad()));
  ConvBackend backend =
      select_conv_backend(input_t, weight_t, bias_t, stride, padding, dilation,
                          transposed, output_padding, groups);

  if (backend == ConvBackend::Cudnn && input_t.data_ptr()) {
    auto input = input_t;
    auto weight = weight_t;
    auto z = z_t;
    auto k = weight.ndimension();
    int64_t dim = k - 2;

    TORCH_CHECK(dim > 0, "weight should have at least three dimensions");

    // NOTE: ConvParams is not a TORCH_API, but its constructor is trivial,
    // so it can be initialized here. Its all other methods cannot be called.
    // This is a dirty hack to avoid code duplication.
    // We need ConvParams to support 3D convolution in CUDNN.

    // NOTE: In PyTorch 2.0, ConvParams is moved from a header file to a cpp
    // file, and we declare it in this file again to use it
    ConvParams params;
    params.stride = expand_param_if_needed(stride, "stride", dim);
    params.padding = expand_param_if_needed(padding, "padding", dim);
    params.dilation = expand_param_if_needed(dilation, "dilation", dim);
    params.transposed = transposed;
    params.output_padding =
        expand_param_if_needed(output_padding, "output_padding", dim);
    params.groups = groups;

    if (k == 3) {
      view1d_as_2d(params);

      // avoid accidentally going through NHWC for permuted 3d input.
      input = input.contiguous();
      input = view4d(input);
      weight = view4d(weight);
      if (z.has_value()) {
        z.emplace(view4d(z.value()));
      }
    }

    Tensor output_t = cudnn_convolution_bias_add_activation(
        input, weight, z, alpha, bias_t, params.stride, params.padding,
        params.dilation, params.groups, activation_mode);
    if (k == 3) {
      output_t = view3d(output_t);
    }
    return output_t;
  }

  Tensor output_t = at::_convolution(input_t, weight_t, bias_t, stride, padding,
                                     dilation, transposed, output_padding,
                                     groups, false, false, true, true);
  if (alpha.has_value() && alpha.value().to<float>() != 0.0 &&
      z_t.has_value()) {
    output_t.add_(z_t.value(), alpha.value());
  }
  switch (activation_mode) {
  case CUDNN_ACTIVATION_IDENTITY:
    break;
  case CUDNN_ACTIVATION_SIGMOID:
    output_t.sigmoid_();
    break;
  case CUDNN_ACTIVATION_RELU:
    output_t.relu_();
    break;
  case CUDNN_ACTIVATION_TANH:
    output_t.tanh_();
    break;
  default:
    TORCH_CHECK(false, "Unsupported activation mode: ", activation_mode);
  }
  return output_t;
}

class CUDNNConvolutionBiasAddActivationWithFallbackFunction
    : public Function<CUDNNConvolutionBiasAddActivationWithFallbackFunction> {
public:
  static Variable
  forward(AutogradContext *ctx, const Variable &input, const Variable &weight,
          const c10::optional<Variable> &bias, const c10::optional<Variable> &z,
          const c10::optional<Scalar> &alpha, IntArrayRef stride,
          IntArrayRef padding, IntArrayRef dilation, bool transposed,
          IntArrayRef output_padding, int64_t groups,
          cudnnActivationMode_t activation_mode) {
    auto output = cudnn_convolution_bias_add_activation_with_fallback_forward(
        input, weight, z, alpha, bias, stride, padding, dilation, transposed,
        output_padding, groups, activation_mode);

    auto z_ = z.has_value() ? bias.value() : Tensor();
    auto bias_ = bias.has_value() ? bias.value() : Tensor();

    ctx->save_for_backward({input, weight, bias_, output});
    ctx->saved_data["alpha"] = alpha;
    ctx->saved_data["stride"] = stride;
    ctx->saved_data["padding"] = padding;
    ctx->saved_data["dilation"] = dilation;
    ctx->saved_data["transposed"] = transposed;
    ctx->saved_data["output_padding"] = output_padding;
    ctx->saved_data["groups"] = groups;
    ctx->saved_data["activation_mode"] = activation_mode;
    ctx->saved_data["z_shape"] =
        z.has_value() ? z.value().sizes() : IntArrayRef();

    return output;
  }

  static variable_list backward(AutogradContext *ctx,
                                const variable_list &grad_output) {
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];
    auto weight = saved[1];
    auto bias = saved[2];
    auto output = saved[3];
    auto alpha = ctx->saved_data["alpha"].toOptional<Scalar>();
    auto stride = ctx->saved_data["stride"].toIntVector();
    auto padding = ctx->saved_data["padding"].toIntVector();
    auto dilation = ctx->saved_data["dilation"].toIntVector();
    auto transposed = ctx->saved_data["transposed"].toBool();
    auto output_padding = ctx->saved_data["output_padding"].toIntVector();
    auto groups = ctx->saved_data["groups"].toInt();
    auto activation_mode = static_cast<cudnnActivationMode_t>(
        ctx->saved_data["activation_mode"].toInt());
    auto z_shape = ctx->saved_data["z_shape"].toIntVector();

    Tensor grad_input, grad_weight, grad_bias, grad_z;
    Tensor act_backward;
    switch (activation_mode) {
    case CUDNN_ACTIVATION_IDENTITY:
      act_backward = grad_output[0];
      break;
    case CUDNN_ACTIVATION_SIGMOID:
      act_backward = sigmoid_backward(grad_output[0], output);
      break;
    case CUDNN_ACTIVATION_RELU:
      act_backward = threshold_backward(grad_output[0], output, 0.0);
      break;
    case CUDNN_ACTIVATION_TANH:
      act_backward = tanh_backward(grad_output[0], output);
      break;
    default:
      TORCH_CHECK(false, "Unsupported activation mode: ", activation_mode);
    }

    if (ctx->needs_input_grad(0) || ctx->needs_input_grad(1) ||
        ctx->needs_input_grad(2)) {
      auto conv_grads = torch::convolution_backward(
          act_backward, input, weight,
          bias.defined() ? bias.sizes() : c10::optional<IntArrayRef>(), stride,
          padding, dilation, transposed, output_padding, groups,
#if 1
          std::array<bool, 3>({ctx->needs_input_grad(0),
                               ctx->needs_input_grad(1),
                               ctx->needs_input_grad(2)})
#else
          std::array<bool, 3>({true, true, true})
#endif
      );
      grad_input = std::get<0>(conv_grads);
      grad_weight = std::get<1>(conv_grads);
      grad_bias = std::get<2>(conv_grads);
    }
    if (!z_shape.empty() &&
#if 1
        ctx->needs_input_grad(3)
#else
        true
#endif
    ) {
      float _alpha = alpha.has_value() ? alpha.value().to<float>() : 1.0f;
      if (_alpha == 0.0f) {
        grad_z = at::zeros(z_shape, act_backward.options());
      } else {
        grad_z = act_backward;
        if (z_shape != grad_z.sizes()) {
          grad_z = at::sum_to(grad_z, z_shape);
        }
        if (_alpha != 1.0f) {
          grad_z = grad_z * _alpha;
        }
      }
    }

    return {grad_input, grad_weight, grad_bias, grad_z,   Tensor(), Tensor(),
            Tensor(),   Tensor(),    Tensor(),  Tensor(), Tensor(), Tensor()};
  }
};

Tensor cudnn_convolution_bias_add_activation_with_fallback(
    const Tensor &input_t, const Tensor &weight_t,
    const c10::optional<Tensor> &bias_t, const c10::optional<Tensor> &z_t,
    const c10::optional<Scalar> &alpha, IntArrayRef stride, IntArrayRef padding,
    IntArrayRef dilation, bool transposed, IntArrayRef output_padding,
    int64_t groups,
    cudnnActivationMode_t activation_mode = CUDNN_ACTIVATION_IDENTITY) {
  return CUDNNConvolutionBiasAddActivationWithFallbackFunction::apply(
      input_t, weight_t, bias_t, z_t, alpha, stride, padding, dilation,
      transposed, output_padding, groups, activation_mode);
}

Tensor cudnn_convolution_bias_add(const Tensor &input_t, const Tensor &weight_t,
                                  const c10::optional<Tensor> &bias_t,
                                  const c10::optional<Tensor> &z_t,
                                  const c10::optional<Scalar> &alpha,
                                  IntArrayRef stride, IntArrayRef padding,
                                  IntArrayRef dilation, bool transposed,
                                  IntArrayRef output_padding, int64_t groups) {
  return cudnn_convolution_bias_add_activation_with_fallback(
      input_t, weight_t, bias_t, z_t, alpha, stride, padding, dilation,
      transposed, output_padding, groups, CUDNN_ACTIVATION_IDENTITY);
}

Tensor cudnn_convolution_bias(const Tensor &input_t, const Tensor &weight_t,
                              const c10::optional<Tensor> &bias_t,
                              IntArrayRef stride, IntArrayRef padding,
                              IntArrayRef dilation, bool transposed,
                              IntArrayRef output_padding, int64_t groups) {
  return cudnn_convolution_bias_add(input_t, weight_t, bias_t, c10::nullopt,
                                    c10::nullopt, stride, padding, dilation,
                                    transposed, output_padding, groups);
}

Tensor cudnn_convolution_bias_sigmoid(const Tensor &input_t,
                                      const Tensor &weight_t,
                                      const c10::optional<Tensor> &bias_t,
                                      IntArrayRef stride, IntArrayRef padding,
                                      IntArrayRef dilation, bool transposed,
                                      IntArrayRef output_padding,
                                      int64_t groups) {
  return cudnn_convolution_bias_add_activation_with_fallback(
      input_t, weight_t, bias_t, c10::nullopt, c10::nullopt, stride, padding,
      dilation, transposed, output_padding, groups, CUDNN_ACTIVATION_SIGMOID);
}

Tensor cudnn_convolution_bias_relu(const Tensor &input_t,
                                   const Tensor &weight_t,
                                   const c10::optional<Tensor> &bias_t,
                                   IntArrayRef stride, IntArrayRef padding,
                                   IntArrayRef dilation, bool transposed,
                                   IntArrayRef output_padding, int64_t groups) {
  return cudnn_convolution_bias_add_activation_with_fallback(
      input_t, weight_t, bias_t, c10::nullopt, c10::nullopt, stride, padding,
      dilation, transposed, output_padding, groups, CUDNN_ACTIVATION_RELU);
}

Tensor cudnn_convolution_bias_tanh(const Tensor &input_t,
                                   const Tensor &weight_t,
                                   const c10::optional<Tensor> &bias_t,
                                   IntArrayRef stride, IntArrayRef padding,
                                   IntArrayRef dilation, bool transposed,
                                   IntArrayRef output_padding, int64_t groups) {
  return cudnn_convolution_bias_add_activation_with_fallback(
      input_t, weight_t, bias_t, c10::nullopt, c10::nullopt, stride, padding,
      dilation, transposed, output_padding, groups, CUDNN_ACTIVATION_TANH);
}

Tensor cudnn_convolution_bias_add_sigmoid(
    const Tensor &input_t, const Tensor &weight_t,
    const c10::optional<Tensor> &bias_t, const c10::optional<Tensor> &z_t,
    const c10::optional<Scalar> &alpha, IntArrayRef stride, IntArrayRef padding,
    IntArrayRef dilation, bool transposed, IntArrayRef output_padding,
    int64_t groups) {
  return cudnn_convolution_bias_add_activation_with_fallback(
      input_t, weight_t, bias_t, z_t, alpha, stride, padding, dilation,
      transposed, output_padding, groups, CUDNN_ACTIVATION_SIGMOID);
}

Tensor cudnn_convolution_bias_add_relu(
    const Tensor &input_t, const Tensor &weight_t,
    const c10::optional<Tensor> &bias_t, const c10::optional<Tensor> &z_t,
    const c10::optional<Scalar> &alpha, IntArrayRef stride, IntArrayRef padding,
    IntArrayRef dilation, bool transposed, IntArrayRef output_padding,
    int64_t groups) {
  return cudnn_convolution_bias_add_activation_with_fallback(
      input_t, weight_t, bias_t, z_t, alpha, stride, padding, dilation,
      transposed, output_padding, groups, CUDNN_ACTIVATION_RELU);
}

Tensor cudnn_convolution_bias_add_tanh(
    const Tensor &input_t, const Tensor &weight_t,
    const c10::optional<Tensor> &bias_t, const c10::optional<Tensor> &z_t,
    const c10::optional<Scalar> &alpha, IntArrayRef stride, IntArrayRef padding,
    IntArrayRef dilation, bool transposed, IntArrayRef output_padding,
    int64_t groups) {
  return cudnn_convolution_bias_add_activation_with_fallback(
      input_t, weight_t, bias_t, z_t, alpha, stride, padding, dilation,
      transposed, output_padding, groups, CUDNN_ACTIVATION_TANH);
}

} // namespace operators
} // namespace sfast
