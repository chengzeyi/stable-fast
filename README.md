# 🚀Stable Fast

[![wheels](https://github.com/chengzeyi/stable-fast/actions/workflows/wheels.yml/badge.svg?branch=main)](https://github.com/chengzeyi/stable-fast/actions/workflows/wheels.yml)
[![Upload Python Package](https://github.com/chengzeyi/stable-fast/actions/workflows/python-publish.yml/badge.svg)](https://github.com/chengzeyi/stable-fast/actions/workflows/python-publish.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chengzeyi/stable-fast-colab/blob/main/stable_fast_colab.ipynb)

[Discord Channel](https://discord.gg/kQFvfzM4SJ)

`stable-fast` achieves SOTA inference performance on __ALL__ kinds of diffuser models.
And unlike `TensorRT` or `AITemplate`, which takes dozens of minutes to compile a model, `stable-fast` only takes a few seconds to compile a model.
`stable-fast` also supports `dynamic shape`, `LoRA` and `ControlNet` out of the box.

[![](https://mermaid.ink/img/pako:eNpFUsGOmzAQ_ZWRpSgXIDYsCXCoVGl76KGXXQ5V13sYYABLYCNsdhNF_HtNqLYHj948j_zezPjOatMQK9jhcFdauQLuR9fTSMcCjq2ZybrjCuvhIPX1Vvc4u7Aih1I75QYCyUo1EvhC6EjTjE7pDowmUCN2BJ_K9fD6DCJKA0hFfPXHAw7W0WQDXwkv5W944hmXzEuEeFUW3iT7QK2GAcGZue4lC7zSA0a1GSc10E59_1nSOA3o_uVeuB3M556UpK2ZX0rIIx6JnbMOq4HCFq2T7F3q2y4o2ejVlKXa6MZKBhzC8BvEnHOppa5whjeR5ZcARCq4jyLNfOQ8CSDPxQYFf2cBG2keUTV-nnepYTO9zVKywsMKLW1Nrr4OF2deb7pmhZsXCtgyNb6JZ4XdjCMrWhzsF_ujUb71L3Iw2JBP78zdpm1znbLOP-mtt6rb-GUePN07N9nidNquo87vYam24Z2sarY19h_5-XSOzxnGCZ0vCaZJ0tSVyLM2fhJtc-EiRrauAZtQ_zHmvyt6-Pm1f5vH71n_Alb0uYg?type=png)](https://mermaid.live/edit#pako:eNpFUsGOmzAQ_ZWRpSgXIDYsCXCoVGl76KGXXQ5V13sYYABLYCNsdhNF_HtNqLYHj948j_zezPjOatMQK9jhcFdauQLuR9fTSMcCjq2ZybrjCuvhIPX1Vvc4u7Aih1I75QYCyUo1EvhC6EjTjE7pDowmUCN2BJ_K9fD6DCJKA0hFfPXHAw7W0WQDXwkv5W944hmXzEuEeFUW3iT7QK2GAcGZue4lC7zSA0a1GSc10E59_1nSOA3o_uVeuB3M556UpK2ZX0rIIx6JnbMOq4HCFq2T7F3q2y4o2ejVlKXa6MZKBhzC8BvEnHOppa5whjeR5ZcARCq4jyLNfOQ8CSDPxQYFf2cBG2keUTV-nnepYTO9zVKywsMKLW1Nrr4OF2deb7pmhZsXCtgyNb6JZ4XdjCMrWhzsF_ujUb71L3Iw2JBP78zdpm1znbLOP-mtt6rb-GUePN07N9nidNquo87vYam24Z2sarY19h_5-XSOzxnGCZ0vCaZJ0tSVyLM2fhJtc-EiRrauAZtQ_zHmvyt6-Pm1f5vH71n_Alb0uYg)

| Framework | torch | torch.compile | AIT  | oneflow | TensorRT | __stable-fast__ |
| --------- | ----- | ------------- | ---- | ------- | -------- | --------------- |
| Time/ms   | 1897  | 1510          | 1158 | 1003    | 991      | __1010__        |

__NOTE__: During benchmarking, `TensorRT` is tested with `static batch size` and `CUDA Graph enabled` while `stable-fast` is running with full dynamic shape.

- [🚀Stable Fast](#stable-fast)
  - [Introduction](#introduction)
    - [What is this?](#what-is-this)
    - [Differences With Other Acceleration Libraries](#differences-with-other-acceleration-libraries)
  - [Installation](#installation)
    - [Install Prebuilt Wheels](#install-prebuilt-wheels)
    - [Install From Source](#install-from-source)
  - [Usage](#usage)
    - [Optimize StableDiffusionPipeline](#optimize-stablediffusionpipeline)
    - [Optimize LCM Pipeline](#optimize-lcm-pipeline)
    - [Dynamically Switch LoRA](#dynamically-switch-lora)
    - [Model Quantization](#model-quantization)
    - [Some Common Methods To Speed Up PyTorch](#some-common-methods-to-speed-up-pytorch)
  - [Performance Comparison](#performance-comparison)
    - [RTX 4080 (512x512, batch size 1, fp16, in WSL2)](#rtx-4080-512x512-batch-size-1-fp16-in-wsl2)
    - [H100](#h100)
    - [A100](#a100)
  - [Compatibility](#compatibility)
  - [Troubleshooting](#troubleshooting)

## Introduction

__NOTE__: `stable-fast` is currently only in beta stage and is prone to be buggy, feel free to try it out and give suggestions!

### What is this?

`stable-fast` is an ultra lightweight inference optimization framework for __HuggingFace Diffusers__ on __NVIDIA GPUs__.
`stable-fast` provides super fast inference optimization by utilizing some key techniques and features:

- __CUDNN Convolution Fusion__: `stable-fast` implements a series of fully-functional and fully-compatible CUDNN convolution fusion operators for all kinds of combinations of `Conv + Bias + Add + Act` computation patterns.
- __Low Precision & Fused GEMM__: `stable-fast` implements a series of fused GEMM operators that compute with `fp16` precision, which is fast than PyTorch's defaults (read & write with `fp16` while compute with `fp32`).
- __Fused Linear GEGLU__: `stable-fast` is able to fuse `GEGLU(x, W, V, b, c) = GELU(xW + b) ⊗ (xV + c)` into one CUDA kernel.
- __NHWC & Fused GroupNorm__: `stable-fast` implements a highly optimized fused NHWC `GroupNorm + GELU` operator with OpenAI's `Triton`, which eliminates the need of memory format permutation operators.
- __Fully Traced Model__: `stable-fast` improves the `torch.jit.trace` interface to make it more proper for tracing complex models. Nearly every part of `StableDiffusionPipeline` can be traced and converted to __TorchScript__. It is more stable than `torch.compile` and has a significantly lower CPU overhead than `torch.compile` and supports __ControlNet__ and __LoRA__.
- __CUDA Graph__: `stable-fast` can capture the `UNet`, `VAE` and `TextEncoder` into CUDA Graph format, which can reduce the CPU overhead when the batch size is small. This implemention also supports dynamic shape.
- __Fused Multihead Attention__: `stable-fast` just uses xformers and makes it compatible with __TorchScript__.

My next goal is to keep `stable-fast` as one of the fastest inference optimization frameworks for `diffusers` and also
provide both speedup and VRAM reduction for `transformers`.
In fact, I already use `stable-fast` to optimize LLMs and achieve a significant speedup.
But I still need to do some work to make it more stable and easy to use and provide a stable user interface.

### Differences With Other Acceleration Libraries

- __Fast__: `stable-fast` is specialy optimized for __HuggingFace Diffusers__. It achieves a high performance across many libraries. And it provides a very fast compilation speed within only a few seconds. It is significantly faster than `torch.compile`, `TensorRT` and `AITemplate` in compilation time.
- __Minimal__: `stable-fast` works as a plugin framework for `PyTorch`. It utilizes existing `PyTorch` functionality and infrastructures and is compatible with other acceleration techniques, as well as popular fine-tuning techniques and deployment solutions.
- __Maximum Compatibility__: `stable-fast` is compatible with all kinds of `HuggingFace Diffusers` and `PyTorch` versions. It is also compatible with `ControlNet` and `LoRA`.

## Installation

__NOTE__: `stable-fast` is currently only tested on `Linux` and `WSL2 in Windows`.
You need to install PyTorch with CUDA support at first (versions from 1.12 to 2.1 are suggested).

I only test `stable-fast` with `torch==2.1.0`, `xformers==0.0.22` and `triton==2.1.0` on `CUDA 12.1` and `Python 3.10`.
Other versions might build and run successfully but that's not guaranteed.

### Install Prebuilt Wheels

Download the wheel corresponding to your system from the [Releases Page](https://github.com/chengzeyi/stable-fast/releases) and install it with `pip3 install <wheel file>`.

Currently both __Linux__ and __Windows__ wheels are available.

```bash
# Change cu121 to your CUDA version and <wheel file> to the path of the wheel file.
# And make sure the wheel file is compatible with your PyTorch version.
pip3 install --index-url https://download.pytorch.org/whl/cu121 'diffusers>=0.19.3' 'xformers>=0.0.20' 'torch>=1.12.0' '<wheel file>'
```

### Install From Source

```bash
# Make sure you have CUDNN/CUBLAS installed.
# https://developer.nvidia.com/cudnn
# https://developer.nvidia.com/cublas

# Install PyTorch with CUDA and other packages at first.
# Windows user: Triton might be not available, you could skip it.
# NOTE: 'wheel' is required or you will meet `No module named 'torch'` error when building.
pip3 install wheel 'torch>=1.12.0' 'diffusers>=0.19.3' 'xformers>=0.0.20' 'triton>=2.1.0'

# (Optional) Makes the build much faster.
pip3 install ninja

# Set TORCH_CUDA_ARCH_LIST if running and building on different GPU types.
# You can also install the latest stable release from PyPI.
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

```python
import time
import torch
from diffusers import (StableDiffusionPipeline,
                       EulerAncestralDiscreteScheduler)
from sfast.compilers.stable_diffusion_pipeline_compiler import (
    compile, CompilationConfig)


def load_model():
    model = StableDiffusionPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5',
        torch_dtype=torch.float16)

    model.scheduler = EulerAncestralDiscreteScheduler.from_config(
        model.scheduler.config)
    model.safety_checker = None
    model.to(torch.device('cuda'))
    return model


model = load_model()

config = CompilationConfig.Default()
# xformers and Triton are suggested for achieving best performance.
try:
    import xformers
    config.enable_xformers = True
except ImportError:
    print('xformers not installed, skip')
try:
    import triton
    config.enable_triton = True
except ImportError:
    print('Triton not installed, skip')
# CUDA Graph is suggested for small batch sizes and small resolutions to reduce CPU overhead.
config.enable_cuda_graph = True

model = compile(model, config)

kwarg_inputs = dict(
    prompt=
    '(masterpiece:1,2), best quality, masterpiece, best detailed face, a beautiful girl',
    height=512,
    width=512,
    num_inference_steps=30,
    num_images_per_prompt=1,
)

# NOTE: Warm it up.
# The initial calls will trigger compilation and might be very slow.
# After that, it should be very fast.
for _ in range(3):
    output_image = model(**kwarg_inputs).images[0]

# Let's see it!
# Note: Progress bar might work incorrectly due to the async nature of CUDA.
begin = time.time()
output_image = model(**kwarg_inputs).images[0]
print(f'Inference time: {time.time() - begin:.3f}s')

# Let's view it in terminal!
from sfast.utils.term_image import print_image

print_image(output_image, max_width=80)
```

Refer to [examples/optimize_stable_diffusion_pipeline.py](examples/optimize_stable_diffusion_pipeline.py) for more details.

You can check this Colab to see how it works on T4 GPU: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chengzeyi/stable-fast-colab/blob/main/stable_fast_colab.ipynb)

### Optimize LCM Pipeline

`stable-fast` is able to optimize the newest `latent consistency model` pipeline and achieve a significant speedup.

Refer to [examples/optimize_lcm_pipeline.py](examples/optimize_lcm_lora.py) for more details about how to optimize normal SD model with LCM LoRA.
Refer to [examples/optimize_lcm_pipeline.py](examples/optimize_lcm_pipeline.py) for more details about how to optimize the standalone LCM model.

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

## Performance Comparison

Performance varies very greatly across different hardware/software/platform/driver configurations.
It is very hard to benchmark accurately. And preparing the environment for benchmarking is also a hard job.
I have tested on some platforms before but the results may still be inaccurate.
Note that when benchmarking, the progress bar showed by `tqdm` may be inaccurate because of the asynchronous nature of CUDA.
To solve this problem, I use `CUDA Event` to measure the speed of iterations per second accurately.

`stable-fast` is expected to work better on newer GPUs and newer CUDA versions.
__On older GPUs, the performance increase might be limited.__
__During benchmarking, the progress bar might work incorrectly because of the asynchronous nature of CUDA.__

### RTX 4080 (512x512, batch size 1, fp16, in WSL2)

This is my personal gaming PC😄. It has a more powerful CPU than those from cloud server providers.

| Framework                                | SD 1.5        | SD XL (1024x1024) | SD 1.5 ControlNet |
| ---------------------------------------- | ------------- | ----------------- | ----------------- |
| Vanilla PyTorch (2.1.0)                  | 29.5 it/s     | 4.6 it/s          | 19.7 it/s         |
| torch.compile (2.1.0, max-autotune)      | 40.0 it/s     | 6.1 it/s          | 20.t it/s         |
| AITemplate                               | 44.2 it/s     |                   |                   |
| OneFlow                                  | 53.6 it/s     |                   |                   |
| AUTO1111 WebUI                           | 17.2 it/s     | 3.6 it/s          |                   |
| AUTO1111 WebUI (with SDPA)               | 24.5 it/s     | 4.3 it/s          |                   |
| TensorRT (AUTO1111 WebUI)                | 40.8 it/s     |                   |                   |
| TensorRT Official Demo                   | 52.6 it/s     |                   |                   |
| __stable-fast (with xformers & Triton)__ | __50.8 it/s__ | __8.5 it/s__      | __36.6 it/s__     |

### H100

Thanks for __@Consceleratus__ and __@harishp__'s help, I have tested speed on H100.

| Framework                                | SD 1.5         | SD XL (1024x1024) | SD 1.5 ControlNet |
| ---------------------------------------- | -------------- | ----------------- | ----------------- |
| Vanilla PyTorch (2.1.0)                  | 54.5 it/s      | 14.9 it/s         | 35.8 it/s         |
| torch.compile (2.1.0, max-autotune)      | 66.0 it/s      | 18.5 it/s         | 40.7 it/s         |
| __stable-fast (with xformers & Triton)__ | __104.6 it/s__ | __21.6 it/s__     | __72.6 it/s__     |

### A100

Thanks for __@SuperSecureHuman__ and __@jon-chuang__'s help, benchmarking on A100 is available now.

| Framework                                | SD 1.5        | SD XL (1024x1024) |
| ---------------------------------------- | ------------- | ----------------- |
| Vanilla PyTorch (2.1.0)                  | 35.6 it/s     | 8.7 it/s          |
| torch.compile (2.1.0, max-autotune)      | 41.9 it/s     | 10.0 it/s         |
| __stable-fast (with xformers & Triton)__ | __61.8 it/s__ | __11.9 it/s__     |

## Compatibility

| Model                               | Supported |
| ----------------------------------- | --------- |
| Hugging Face Diffusers (1.5/2.1/XL) | Yes       |
| With ControlNet                     | Yes       |
| With LoRA                           | Yes       |
| Latent Consistency Model            | Yes       |
| SDXL Turbo                          | Yes       |

| Functionality                       | Supported |
| ----------------------------------- | --------- |
| Dynamic Shape                       | Yes       |
| Text to Image                       | Yes       |
| Image to Image                      | Yes       |
| Image Inpainting                    | Yes       |

| UI Framework                        | Supported | Link                                                                    |
| ----------------------------------- | --------- | ----------------------------------------------------------------------- |
| AUTOMATIC1111                       | WIP       |                                                                         |
| SD Next                             | Yes       | [`SD Next`](https://github.com/vladmandic/automatic)                    |
| ComfyUI                             | Yes       | [`ComfyUI_stable_fast`](https://github.com/gameltb/ComfyUI_stable_fast) |

| Operating System                    | Supported |
| ----------------------------------- | --------- |
| Linux                               | Yes       |
| Windows                             | Yes       |
| Windows WSL                         | Yes       |

## Troubleshooting

Refer to [doc/troubleshooting.md](doc/troubleshooting.md) for more details.

And you can join the [Discord Channel](https://discord.gg/kQFvfzM4SJ) to ask for help.
