# 🚀Stable Fast

[![wheels](https://github.com/chengzeyi/stable-fast/actions/workflows/wheels.yml/badge.svg?branch=main)](https://github.com/chengzeyi/stable-fast/actions/workflows/wheels.yml)
[![Upload Python Package](https://github.com/chengzeyi/stable-fast/actions/workflows/python-publish.yml/badge.svg)](https://github.com/chengzeyi/stable-fast/actions/workflows/python-publish.yml)

- [🚀Stable Fast](#stable-fast)
  - [Introduction](#introduction)
    - [What is this?](#what-is-this)
    - [Differences With Other Acceleration Libraries](#differences-with-other-acceleration-libraries)
    - [Performance Comparison](#performance-comparison)
      - [RTX 4080 (512x512, batch size 1, fp16, tcmalloc enabled, in WSL2)](#rtx-4080-512x512-batch-size-1-fp16-tcmalloc-enabled-in-wsl2)
      - [RTX 4090 (512x512, batch size 1, fp16, tcmalloc enabled)](#rtx-4090-512x512-batch-size-1-fp16-tcmalloc-enabled)
      - [RTX 3080 Ti (512x512, batch size 1, fp16, tcmalloc enabled)](#rtx-3080-ti-512x512-batch-size-1-fp16-tcmalloc-enabled)
      - [RTX 3090 (512x512, batch size 1, fp16, tcmalloc enabled)](#rtx-3090-512x512-batch-size-1-fp16-tcmalloc-enabled)
      - [H100](#h100)
      - [A100 PCIe 40GB](#a100-pcie-40gb)
    - [Compatibility](#compatibility)
  - [Installation](#installation)
    - [Install Prebuilt Wheels](#install-prebuilt-wheels)
    - [Install From Source](#install-from-source)
  - [Usage](#usage)
    - [Optimize StableDiffusionPipeline](#optimize-stablediffusionpipeline)
    - [Optimize LCM Pipeline](#optimize-lcm-pipeline)
    - [Dynamically Switch LoRA](#dynamically-switch-lora)
    - [Model Quantization](#model-quantization)
    - [Some Common Methods To Speed Up PyTorch](#some-common-methods-to-speed-up-pytorch)
  - [Troubleshooting](#troubleshooting)

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

My next goal is to keep `stable-fast` as one of the fastest inference optimization frameworks for `diffusers` and also
provide both speedup and VRAM reduction for `transformers`.
In fact, I already use `stable-fast` to optimize LLMs and achieve a significant speedup.
But I still need to do some work to make it more stable and easy to use and provide a stable user interface.

### Differences With Other Acceleration Libraries

- __Fast__: `stable-fast` is specialy optimized for __HuggingFace Diffusers__. It achieves a high performance across many libraries. And it provides a very fast compilation speed within only a few seconds. It is significantly faster than `torch.compile`, `TensorRT` and `AITemplate` in compilation time.
- __Minimal__: `stable-fast` works as a plugin framework for `PyTorch`. It utilizes existing `PyTorch` functionality and infrastructures and is compatible with other acceleration techniques, as well as popular fine-tuning techniques and deployment solutions.
- __Maximum Compatibility__: `stable-fast` is compatible with all kinds of `HuggingFace Diffusers` and `PyTorch` versions. It is also compatible with `ControlNet` and `LoRA`.

### Performance Comparison

Performance varies very greatly across different hardware/software/platform/driver configurations.
It is very hard to benchmark accurately. And preparing the environment for benchmarking is also a hard job.
I have tested on some platforms before but the results may still be inaccurate.
Note that when benchmarking, the progress bar showed by `tqdm` may be inaccurate because of the asynchronous nature of CUDA.
To solve this problem, I have to add `torch.cuda.synchronize()` after every inference step, which will slow down the inference,
so the results might not be very accurate and might be slower than the actual performance.

`stable-fast` is expected to work better on newer GPUs and newer CUDA versions.
__On older GPUs, the performance increase might be limited.__
__During benchmarking, the progress bar might work incorrectly because of the asynchronous nature of CUDA.__

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

__IMPORTANT__

My latest benchmarks show that, on my 4080 machine, for SD 1.5, 512x512, 20 steps, EulerA and TinyVAE:

With `stable-fast` I get `426ms` to finish one image.

With `TensorRT 9.0.1 with static batch and CUDA Graph` I get `425ms` to finish one image. [demo](https://github.com/NVIDIA/TensorRT/blob/5f422623e7f5bdc593b781695cbddda99124c9b8/demo/Diffusion/demo_txt2img.py)

__So `stable-fast` is on par with TensorRT in terms of speed and provides more flexibility and compatibility and is totally open sourced!!!.__

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

#### H100

Thanks for __@Consceleratus__'s help, I have tested speed on H100.

Detailed benchmarking results will be available soon.

#### A100 PCIe 40GB

Thanks for __@SuperSecureHuman__'s help, benchmarking on A100 PCIe 40GB is available now.

| Framework                                | SD 1.5        | SD 2.1         | SD 1.5 ControlNet | SD XL         |
| ---------------------------------------- | ------------- | -------------- | ----------------- | --------------|
| Vanilla PyTorch (2.1.0+cu118)            | 23.8 it/s     | 23.8 it/s      | 15.7 it/s         | 10.0 it/s     |
| torch.compile (2.1.0+cu118, NHWC UNet)   | 37.7 it/s     | 42.7 it/s      | 24.7 it/s         | 20.9 it/s     |
| __Stable Fast (with xformers & Triton)__ | __53.2 it/s__ | __55.9 it/s__  | __37.1 it/s__     | __29.6 it/s__ |

### Compatibility

| Model                               | Supported |
| ----------------------------------- | --------- |
| Hugging Face Diffusers (1.5/2.1/XL) | Yes       |
| With ControlNet                     | Yes       |
| With LoRA                           | Yes       |
| Dynamic Shape                       | Yes       |
| Latent Consistency Model            | Yes       |

| UI Framework                        | Supported | Link                                                                    |
| ----------------------------------- | --------- | ----------------------------------------------------------------------- |
| AUTOMATIC1111                       | WIP       |                                                                         |
| SD Next                             | Yes       | [`SD Next`](https://github.com/vladmandic/automatic)                    |
| ComfyUI                             | Yes       | [`ComfyUI_stable_fast`](https://github.com/gameltb/ComfyUI_stable_fast) |

## Installation

__NOTE__: `stable-fast` is currently only tested on `Linux` and `WSL2 in Windows`.
You need to install PyTorch with CUDA support at first (versions from 1.12 to 2.1 are suggested).

I only test `stable-fast` with `torch==2.1.0`, `xformers==0.0.22` and `triton==2.1.0` on `CUDA 12.1` and `Python 3.10`.
Other versions might build and run successfully but that's not guaranteed.

### Install Prebuilt Wheels

Download the wheel corresponding to your system from the [Releases Page](https://github.com/chengzeyi/stable-fast/releases) and install it with `pip3 install <wheel file>`.

Currently both __Linux__ and __Windows__ wheels are available.

Linux

```bash
# Linux
pip3 install 'diffusers>=0.19.3' 'xformers>=0.0.20' 'triton>=2.1.0' 'torch>=1.12.0' <wheel file>
```

Windows

```powershell
# Change cu121 to your CUDA version
pip3 install 'diffusers>=0.19.3' 'xformers>=0.0.20' 'torch>=1.12.0' <wheel file> --index-url https://download.pytorch.org/whl/cu121
```

### Install From Source

```bash
# Make sure you have CUDNN/CUBLAS installed.
# https://developer.nvidia.com/cudnn
# https://developer.nvidia.com/cublas

# Install PyTorch with CUDA and other packages at first
pip3 install 'torch>=1.12.0' 'diffusers>=0.19.3' 'xformers>=0.0.20' 'triton>=2.1.0'
# Windows user: Triton might be not available, you could skip it.

# (Optional) Makes the build much faster
pip3 install ninja

# Set TORCH_CUDA_ARCH_LIST if running and building on different GPU types
# You can also install the latest stable release from PyPI
# pip3 install -v -U stable-fast
pip3 install -v -U git+https://github.com/chengzeyi/stable-fast.git@main#egg=stable-fast
# (this can take dozens of minutes)
```

__NOTE__: Any usage outside `sfast.compilers` is not guaranteed to be backward compatible.

__NOTE__: To get the best performance, `xformers` and OpenAI's `triton>=2.1.0` need to be installed and enabled.
You might need to build `xformers` from source to make it compatible with your `PyTorch`.

## Usage

### Optimize StableDiffusionPipeline

`stable-fast` is able to optimize `StableDiffusionPipeline` and `StableDiffusionPipelineXL` directly.

Refer to [examples/optimize_stable_diffusion_pipeline.py](examples/optimize_stable_diffusion_pipeline.py) for more details.

### Optimize LCM Pipeline

`stable-fast` is able to optimize the newest `latent consistency model` pipeline and achieve a significant speedup.

Refer to [examples/optimize_lcm_pipeline.py](examples/optimize_lcm_pipeline.py) for more details.

### Dynamically Switch LoRA

Switching LoRA dynamically is supported but you need to do some extra work.
It is possible because the compiled graph and `CUDA Graph` share the same
underlaying data (pointers) with the original UNet model. So all you need to do
is to update the original UNet model's parameters inplace.

The following code assumes you have already load a LoRA and compiled the model,
and you want to switch to another LoRA.

```python
# load_state_dict with assign=True requires torch >= 2.1.0

def update_state_dict(dst, src):
    for key, value in src.items():
        # Do inplace copy.
        # As the traced forward function shares the same underlaying data (pointers),
        # this modification will be reflected in the traced forward function.
        dst[key].copy_(value)

# Switch "another" LoRA into UNet
def switch_lora(unet, lora):
    # Store the original UNet parameters
    state_dict = unet.state_dict()
    # Load another LoRA into unet
    unet.load_attn_procs(lora)
    # Inplace copy current UNet parameters to the original unet parameters
    update_state_dict(state_dict, unet.state_dict())
    # Load the original UNet parameters back.
    # We use assign=True because we still want to hold the references
    # of the original UNet parameters
    unet.load_state_dict(state_dict, assign=True)

switch_lora(compiled_model.unet, lora_b_path)
```

### Model Quantization

`stable-fast` extends PyTorch's `quantize_dynamic` functionality and provides a fast quantized linear operator.
By enabling it, you could get a slight VRAM reduction for `diffusers` and significant VRAM reduction for `transformers`,
and cound get a potential speedup.

However, since `diffusers` implements its own `Linear` layer as `LoRACompatibleLinear`,
you need to do some hacks to make it work and it is a little complex and tricky.

Refer to [tests/compilers/test_stable_diffusion_pipeline_compiler.py](tests/compilers/test_stable_diffusion_pipeline_compiler.py) to see how to do it.

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

## Troubleshooting

Refer to [doc/troubleshooting.md](doc/troubleshooting.md) for more details.
