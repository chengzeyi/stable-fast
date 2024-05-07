#include <torch/extension.h>

#include <cstddef>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/macros/Macros.h>

namespace sfast {
namespace operators {
namespace lookup_activation {

namespace {

template <template <typename> class Act> struct FullRangeAct {
  template <typename T> struct Initializer {
    C10_HOST_DEVICE
    T operator()(size_t idx) const {
      T x;
      if constexpr (sizeof(T) == 1) {
        uint8_t val = static_cast<uint8_t>(idx);
        x = *reinterpret_cast<T *>(&val);
      } else if constexpr (sizeof(T) == 2) {
        uint16_t val = static_cast<uint16_t>(idx);
        x = *reinterpret_cast<T *>(&val);
      } else if constexpr (sizeof(T) == 4) {
        uint32_t val = static_cast<uint32_t>(idx);
        x = *reinterpret_cast<T *>(&val);
      } else if constexpr (sizeof(T) == 8) {
        uint64_t val = static_cast<uint64_t>(idx);
        x = *reinterpret_cast<T *>(&val);
      }
      Act<T> act;
      return act(x);
    }
  };
};

template <typename T> struct FullRangeActLookupper {
  C10_HOST_DEVICE
  FullRangeActLookupper(const T *data) : data_(data) {}

  C10_HOST_DEVICE
  T operator()(T val) const {
    if constexpr (sizeof(T) == 1) {
      uint8_t idx = *reinterpret_cast<const uint8_t *>(&val);
      return data_[idx];
    } else if constexpr (sizeof(T) == 2) {
      uint16_t idx = *reinterpret_cast<const uint16_t *>(&val);
      return data_[idx];
    } else if constexpr (sizeof(T) == 4) {
      uint32_t idx = *reinterpret_cast<const uint32_t *>(&val);
      return data_[idx];
    } else if constexpr (sizeof(T) == 8) {
      uint64_t idx = *reinterpret_cast<const uint64_t *>(&val);
      return data_[idx];
    } else {
      return 0;
    }
  }

  const T *data_;
};

template <typename T> inline size_t get_size_in_bytes(size_t size) {
  return size * sizeof(T);
}

template <typename T> inline size_t get_full_range_elements() {
  return 1 << (sizeof(T) * 8);
}

template <typename T> inline size_t get_full_range_size_in_bytes() {
  return get_size_in_bytes<T>(get_full_range_elements<T>());
}

template <typename T, template <typename> class Initializer>
__global__ void init_data_kernel(T *data, size_t size) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  Initializer<T> initializer;
  if (idx < size) {
    data[idx] = initializer(idx);
  }
}

template <typename T, template <typename> class Initializer>
void launch_init_data_kernel(T *data, size_t size, int block_size = 256) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  int grid_size = (size + block_size - 1) / block_size;
  init_data_kernel<T, Initializer>
      <<<grid_size, block_size, 0, stream>>>(data, size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T, template <typename> class Act>
void launch_init_full_range_data_kernel(T *data, int block_size = 256) {
  launch_init_data_kernel<T, FullRangeAct<Act>::template Initializer>(
      data, get_full_range_elements<T>(), block_size);
}

bool set_persistent_cache(cudaStream_t stream, const void *data, size_t size) {
  size_t persistent_l2_cache_size_limit;
  C10_CUDA_CHECK(cudaDeviceGetLimit(&persistent_l2_cache_size_limit,
                                    cudaLimitPersistingL2CacheSize));
  if (size > persistent_l2_cache_size_limit) {
    cudaError_t err = cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size);
    if (err) {
      TORCH_WARN_ONCE("Failed to set cudaLimitPersistingL2CacheSize to ", size,
                      " bytes: ", cudaGetErrorString(err));
      return false;
    }
  }

  cudaStreamAttrValue stream_attribute;
  stream_attribute.accessPolicyWindow.base_ptr = const_cast<void *>(data);
  stream_attribute.accessPolicyWindow.num_bytes = size;
  stream_attribute.accessPolicyWindow.hitRatio = 1.0f;
  stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
  stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyNormal;

  C10_CUDA_CHECK(cudaStreamSetAttribute(
      stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute));
  return true;
}

bool unset_persistent_cache(cudaStream_t stream) {
  cudaStreamAttrValue stream_attribute;
  stream_attribute.accessPolicyWindow.base_ptr = nullptr;
  stream_attribute.accessPolicyWindow.num_bytes = 0;
  stream_attribute.accessPolicyWindow.hitRatio = 0.0f;
  stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyNormal;
  stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyNormal;

  C10_CUDA_CHECK(cudaStreamSetAttribute(
      stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute));
  C10_CUDA_CHECK(cudaCtxResetPersistingL2Cache());
  return true;
}

struct PersistentCacheGuard {
  PersistentCacheGuard(cudaStream_t stream, const void *data, size_t size)
      : stream_(stream) {
    if (set_persistent_cache(stream_, data, size)) {
      succ_ = true;
    }
  }

  ~PersistentCacheGuard() {
    if (succ_) {
      TORCH_CHECK(unset_persistent_cache(stream_));
    }
  }

  bool succ() const { return succ_; }

  cudaStream_t stream_;
  bool succ_ = false;
};

} // namespace

} // namespace lookup_activation
} // namespace operators
} // namespace sfast
