# 🚀Stable Fast

## Introduction

__NOTE__: `stable-fast` is currently only in beta stage and is prone to be buggy, feel free to try it out and give suggestions!

### What is this?

`stable-fast` is an ultra lightweight inference optimization framework for __HuggingFace Diffusers__ on __NVIDIA GPUs__.
`stable-fast` provides super fast inference optimization by utilizing some key techniques and features:

- __CUDNN Convolution Fusion__: `stable-fast` implements a series of fully-functional and fully-compatible CUDNN convolution fusion operators for all kinds of combinations of `Conv + Bias + Add + Act` computation patterns.
- __Low Precision & Fused GEMM__: `stable-fast` implements a series of fused GEMM operators that compute with `fp16` precision, which is fast than PyTorch's defaults (read & write with `fp16` while compute with `fp32`).
- __NHWC & Fused GroupNorm__: `stable-fast` implements a highly optimized fused NHWC `GroupNorm + GELU` operator with OpenAI's `Triton`, which eliminates the need of memory format permutation operators.
- __Fully Traced Model__: `stable-fast` improves the `torch.jit.trace` interface to make it more proper for tracing complex models. Nearly every part of `StableDiffusionPipeline` can be traced and converted to __TorchScript__. It is more stable than `torch.compile` and has a significantly lower CPU overhead than `torch.compile` and supports __ControlNet__ and __LoRA__.
- __CUDA Graph__: `stable-fast` can capture the UNet structure into CUDA Graph format, which can reduce the CPU overhead when the batch size is small.
- __Fused Multihead Attention__: `stable-fast` just uses xformers and make it compatible with __TorchScript__.

### Differences With Other Acceleration Libraries

- __Fast__: `stable-fast` is specialy optimized for __HuggingFace Diffusers__. It achieves a high performance across many libraries.
- __Minimal__: `stable-fast` works as a plugin framework for `PyTorch`. It utilizes existing `PyTorch` functionality and infrastructures and is compatible with other acceleration techniques, as well as popular fine-tuning techniques and deployment solutions.

### Performance Comparison

Performance varies very greatly across different hardware/software/platform/driver configurations.
It is very hard to benchmark accurately. And preparing the environment for benchmarking is also a hard job.
I have tested on some platforms before but the results may still be inaccurate.

#### RTX 4080 (512x512, batch size 1, fp16, tcmalloc enabled, in WSL2)

This is my personal gaming PC😄. It has a more powerful CPU than those from cloud server providers.

| Framework                                | SD 1.5        | SD 2.1        | SD XL (1024x1024) |
| ---------------------------------------- | ------------- | ------------- | ----------------- |
| Vanilla PyTorch (2.1.0+cu118)            | 29.5 it/s     | 32.4 it/s     | 4.6 it/s          |
| torch.compile (2.1.0+cu118, NHWC UNet)   | 40.0 it/s     | 44.0 it/s     | 6.1 it/s          |
| AITemplate                               | 44.2 it/s     | untested      | untested          |
| OneFlow                                  | 50.3 it/s     | untested      | untested          |
| AUTO1111 WebUI                           | 17.2 it/s     | 15.2 it/s     | 3.6 it/s          |
| AUTO1111 WebUI (with SDPA)               | 24.5 it/s     | 26.1 it/s     | 4.3 it/s          |
| TensorRT (AUTO1111 WebUI)                | 40.8 it/s     | untested      | untested          |
| __Stable Fast (with xformers & Triton)__ | __49.7 it/s__ | __52.5 it/s__ | __8.1 it/s__      |

#### RTX 4090 (512x512, batch size 1, fp16, tcmalloc enabled)

| Framework                                | SD 1.5        | SD 2.1         | SD 1.5 ControlNet |
| ---------------------------------------- | ------------- | -------------- | ----------------- |
| Vanilla PyTorch (2.1.0+cu118)            | 24.9 it/s     | 27.1 it/s      | 18.9 it/s         |
| torch.compile (2.1.0+cu118, NHWC UNet)   | 33.5 it/s     | 38.2 it/s      | 22.7 it/s         |
| AITemplate                               | 65.7 it/s     | 71.6 it/s      | untested          |
| OneFlow                                  | 60.1 it/s     | 12.9 it/s (??) | untested          |
| TensorRT                                 | untested      | untested       | untested          |
| __Stable Fast (with xformers & Triton)__ | __61.8 it/s__ | __61.6 it/s__  | __42.3 it/s__     |

(??): OneFlow seems to be not working well with SD 2.1

#### RTX 3080 Ti (512x512, batch size 1, fp16, tcmalloc enabled)

| Framework                                | SD 1.5        | SD 2.1         | SD 1.5 ControlNet |
| ---------------------------------------- | ------------- | -------------- | ----------------- |
| Vanilla PyTorch (2.1.0+cu118)            | 19.3 it/s     | 20.4 it/s      | 13.8 it/s         |
| torch.compile (2.1.0+cu118, NHWC UNet)   | 24.4 it/s     | 26.9 it/s      | 17.7 it/s         |
| AITemplate                               | untested      | untested       | untested          |
| OneFlow                                  | 32.8 it/s     | 8.82 it/s (??) | untested          |
| TensorRT                                 | untested      | untested       | untested          |
| __Stable Fast (with xformers & Triton)__ | __28.1 it/s__ | __30.2 it/s__  | __20.0 it/s__     |

(??): OneFlow seems to be not working well with SD 2.1

#### RTX 3090 (512x512, batch size 1, fp16, tcmalloc enabled)

| Framework                                | SD 1.5        |
| ---------------------------------------- | ------------- |
| Vanilla PyTorch (2.1.0+cu118)            | 22.5 it/s     |
| torch.compile (2.1.0+cu118, NHWC UNet)   | 25.3 it/s     |
| AITemplate                               | 34.6 it/s     |
| OneFlow                                  | 38.8 it/s     |
| TensorRT                                 | untested      |
| __Stable Fast (with xformers & Triton)__ | __31.5 it/s__ |

#### A100

Sorry, currently A100 is hard and expensive to rent from cloud server providers in my region.
A few months ago I have tested this framework on A100 and the speed is around __61 it/s__ for SD 1.5.
Detailed benchmark results will be available when I have the access to A100 again.

### Compatibility

| Model                               | Supported |
| ----------------------------------- | --------- |
| Hugging Face Diffusers (1.5/2.1/XL) | Yes       |
| With ControlNet                     | Yes       |
| With LoRA                           | Yes       |
| Dynamic Shape                       | Yes       |

| UI Framework                        | Supported | Link                                                                    |
| ----------------------------------- | --------- | ----------------------------------------------------------------------- |
| AUTOMATIC1111                       | WIP       |                                                                         |
| SD Next                             | WIP       |                                                                         |
| ComfyUI                             | Yes       | [`ComfyUI_stable_fast`](https://github.com/gameltb/ComfyUI_stable_fast) |

## Usage

### Optimize StableDiffusionPipeline

```python
import torch
from diffusers import (StableDiffusionPipeline, EulerAncestralDiscreteScheduler)
from sfast.compilers.stable_diffusion_pipeline_compiler import (compile,
                                                                CompilationConfig
                                                                )

def load_model():
    # NOTE:
    # You could change to StableDiffusionXLPipeline to load SDXL model.
    # If the resolution is high (1024x1024),
    # ensure you VRAM is sufficient, especially when you are on Windows or WSL,
    # where the GPU driver may choose to allocate from "shared VRAM" when OOM would occur.
    # Or the performance might regress.
    # from diffusers import StableDiffusionXLPipeline
    #
    # model = StableDiffusionXLPipeline.from_pretrained(
    #     'stabilityai/stable-diffusion-xl-base-1.0', torch_dtype=torch.float16)

    model = StableDiffusionPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', torch_dtype=torch.float16)

    model.scheduler = EulerAncestralDiscreteScheduler.from_config(
        model.scheduler.config)
    model.safety_checker = None
    model.to(torch.device('cuda'))
    return model

model = load_model()

config = CompilationConfig.Default()

# xformers and Triton are suggested for achieving best performance.
# It might be slow for Triton to generate, compile and fine-tune kernels.
try:
    import xformers
    config.enable_xformers = True
except ImportError:
    print('xformers not installed, skip')
# NOTE:
# When GPU VRAM is insufficient or the architecture is too old, Triton might be slow.
# Disable Triton if you encounter this problem.
try:
    import triton
    config.enable_triton = True
except ImportError:
    print('Triton not installed, skip')
# NOTE:
# CUDA Graph is suggested for small batch sizes and small resolutions to reduce CPU overhead.
# My implementation can handle dynamic shape with increased need for GPU memory.
# But when your GPU VRAM is insufficient or the image resolution is high,
# CUDA Graph could cause less efficient VRAM utilization and slow down the inference,
# especially when on Windows or WSL which has the "shared VRAM" mechanism.
# If you meet problems related to it, you should disable it.
config.enable_cuda_graph = True

compiled_model = compile(model, config)

kwarg_inputs = dict(
    prompt=
    '(masterpiece:1,2), best quality, masterpiece, best detail face, lineart, monochrome, a beautiful girl',
    # NOTE: If you use SDXL, you should use a higher resolution to improve the generation quality.
    height=512,
    width=512,
    num_inference_steps=30,
    num_images_per_prompt=1,
)

# NOTE: Warm it up.
# The first call will trigger compilation and might be very slow.
# After the first call, it should be very fast.
output_image = compiled_model(**kwarg_inputs).images[0]

# Let's see the second call!
output_image = compiled_model(**kwarg_inputs).images[0]
```

### Installation

__NOTE__: `stable-fast` is currently only tested on `Linux` and `WSL2 in Windows`.
You need to install PyTorch with CUDA support at first (versions from 1.12 to 2.1 are suggested).
I only test `stable-fast` with `torch==2.1.0`, `xformers==0.0.22` and `triton==2.1.0` on `CUDA 12.1`.
Other versions might build and run successfully but that's not guaranteed.

#### Install From Source

```bash
# Make sure you have CUDNN/CUBLAS installed.
# https://developer.nvidia.com/cudnn
# https://developer.nvidia.com/cublas

# Install PyTorch with CUDA and other packages at first
pip3 install 'torch>=1.12.0' 'diffusers>=0.19.3' 'xformers>=0.0.20' 'triton>=2.1.0'

# (Optional) Makes the build much faster
pip3 install ninja

# Set TORCH_CUDA_ARCH_LIST if running and building on different GPU types
pip3 install -v -U git+https://github.com/chengzeyi/stable-fast.git@main#egg=stable-fast
# (this can take dozens of minutes)
```

__NOTE__: Any usage outside `sfast.compilers` is not guaranteed to be backward compatible.

__NOTE__: To get the best performance, `xformers` and OpenAI's `triton>=2.1.0` need to be installed and enabled.
You might need to build `xformers` from source to make it compatible with your `PyTorch`.

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

## Trouble Shooting

### Compilation Is SO SLOW. How To Improve It?

Dynamic code generation is usually the cause for slow compilation.
You could disable features related to it to speed up compilation.
But this might slow down your inference.

#### Disable JIT Optimized Execution

```python
# Wrap your code in this context manager
with torch.jit.optimized_execution(False):
    # Do your things
```

#### Disable Triton

```python
config.enable_triton = False
```

### Inference Is SO SLOW. What's Wrong?

#### Disable CUDA Graph When GPU VRAM Is Insufficient

When your GPU VRAM is insufficient or the image resolution is high,
CUDA Graph could cause less efficient VRAM utilization and slow down the inference.

```python
config.enable_cuda_graph = False
```
