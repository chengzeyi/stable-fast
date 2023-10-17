# Stable Fast

## Introduction

__NOTE: `stable-fast` is only in beta stage and is prone to be buggy, feel free to try it out and give suggestions!__

### What is this?

`stable-fast` is an ultra lightweight inference optimization library for __HuggingFace Diffusers__ on __NVIDIA GPUs__.
`stable-fast` provides super fast inference optimization by utilizing some key techniques and features:

- __CUDNN Convolution Fusion__: `stable-fast` implements a series of fully-functional and fully-compatible CUDNN convolution fusion operators for all kinds of combinations of `Conv + Bias + Add + Act` computation patterns.
- __Low Precision & Fused GEMM__: `stable-fast` implements a series of fused GEMM operators that compute with `fp16` precision, which is fast than PyTorch's defaults (read & write with `fp16` while compute with `fp32`).
- __NHWC & Fused GroupNorm__: `stable-fast` implements a highly optimized fused NHWC `GroupNorm + GELU` operator with OpenAI's `triton`, which eliminates the need of memory format permutation operators.
- __Fully Traced Model__: `stable-fast` improves the `torch.jit.trace` interface to make it more proper for tracing complex models. Nearly every part of `StableDiffusionPipeline` can be traced and converted to __TorchScript__. It is more stable than `torch.compile` and has a significantly lower CPU overhead than `torch.compile` and supports __ControlNet__ and __LoRA__.
- __CUDA Graph__: `stable-fast` can capture the UNet structure into CUDA Graph format, which can reduce the CPU overhead when the batch size is small.
- __Fused Multihead Attention__: `stable-fast` just uses xformers and make it compatible with __TorchScript__.

### Differences With Other Acceleration Libraries

- __Fast__: `stable-fast` is specialy optimized for __HuggingFace Diffusers__. It achieves the best performance over all libraries.
- __Minimal__: `stable-fast` works as a plugin framework for `PyTorch`. it utilizes existing `PyTorch` functionality and infrastructures and is compatible with other acceleration techniques, as well as popular fine-tuning techniques and deployment solutions.

### Performance Comparison

#### A100 SXM 80GB (SD v1.5, 512x512, fp16)

| Framework                                | Performance |
| ---------------------------------------- | ----------- |
| Vanilla PyTorch                          | 23 it/s     |
| AITemplate                               | 44 it/s     |
| TensorRT                                 | 52 it/s     |
| OneFlow                                  | 55 it/s     |
| __Stable Fast (with xformers & triton)__ | __60 it/s__ |

#### RTX 3090 Ti (SD v1.5, 512x512, fp16)

| Framework                                | Performance |
| ---------------------------------------- | ----------- |
| Vanilla PyTorch                          | 16 it/s     |
| AITemplate                               | 31 it/s     |
| TensorRT                                 | 33 it/s     |
| OneFlow                                  | 39 it/s     |
| __Stable Fast (with xformers & triton)__ | __38 it/s__ |

## Usage

### Installation

__NOTE: `stable-fast` is currently only tested on Linux. You need to install PyTorch with CUDA support at first (versions from 1.12 to 2.1 are suggested).__

#### Install From Source

```bash
# Make sure you have CUDNN/CUBLAS installed.
# https://developer.nvidia.com/cudnn
# https://developer.nvidia.com/cublas

# Install PyTorch with CUDA and other packages at first
pip install torch diffusers xformers triton

# (Optional) Makes the build much faster
pip install ninja

# Set TORCH_CUDA_ARCH_LIST if running and building on different GPU types
pip install -v -U git+https://github.com/chengzeyi/stable-fast.git@main#egg=stable-fast
# (this can take dozens of minutes)
```

__NOTE: Any usage outside `sfast.compilers` is not guaranteed to be backward compatible.__
__NOTE: To get the best performance, `xformers` and OpenAI's `triton>=2.1.0` need to be installed and enabled. You might need to build `xformers` from source to make it compatible with your `PyTorch`.__

### Some Common Methods To Speed Up PyTorch

```bash
# TCMalloc is highly suggested to reduce CPU overhead
# https://github.com/google/tcmalloc
LD_PRELOAD=/path/to/libtcmalloc.so python3 ...
```

```python
import packaging.version
import torch

if packaging.version.parse(torch.__version__) >= packaging.version.parse('1.12.0'):
    torch.backends.cuda.matmul.allow_tf32 = True
```

### Optimize StableDiffusionPipeline

```python
import torch
from diffusers import StableDiffusionPipeline
from sfast.compilers.stable_diffusion_pipeline_compiler import (compile,
                                                                CompilationConfig
                                                                )

def load_model():
    model = StableDiffusionPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', torch_dtype=torch.float16)
    model.safety_checker = None
    model.to(torch.device('cuda'))
    return model

model = load_model()

config = CompilationConfig.Default()
# xformers and triton are suggested for achieving best performance.
# It might be slow for triton to generate, compile and fine-tune kernels.
try:
    import xformers
    config.enable_xformers = True
except ImportError:
    print('xformers not installed, skip')
try:
    import triton
    config.enable_triton = True
except ImportError:
    print('triton not installed, skip')
# CUDA Graph is suggested for small batch sizes.
# After capturing, the model only accepts one fixed image size.
# If you want the model to be dynamic, don't enable it.
config.enable_cuda_graph = True
compiled_model = compile(model, config)

kwarg_inputs = dict(
    prompt=
    '(masterpiece:1,2), best quality, masterpiece, best detail face, lineart, monochrome, a beautiful girl',
    height=512,
    width=512,
    num_inference_steps=50,
    num_images_per_prompt=1,
)

# NOTE: Warm it up.
# The first call will trigger compilation and might be very slow.
# After the first call, it should be very fast.
output_image = compiled_model(**kwarg_inputs).images[0]

# Let's see the second call!
output_image = compiled_model(**kwarg_inputs).images[0]
```
