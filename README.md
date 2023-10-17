# Stable Fast

## Introduction

### What is this?

`stable-fast` is an ultra lightweight inference optimization library for __HuggingFace Diffusers__ on __NVIDIA GPUs__.
`stable-fast` provides super fast inference optimization by utilizing some key techniques and features:

- __CUDNN Convolution Fusion__: `stable-fast` implements a series of fully-functional and fully-compatible CUDNN convolution fusion operators for all kinds of combinations of `Conv + Bias + Add + Act` computation patterns.
- __Low Precision & Fused GEMM__: `stable-fast` implements a series of fused GEMM operators that compute with `fp16` precision, which is fast than PyTorch's defaults (read & write with `fp16` while compute with `fp32`).
- __NHWC & Fused GroupNorm__: `stable-fast` implements a highly optimized fused NHWC `GroupNorm + GELU` operator with OpenAI's `triton`, which eliminates the need of memory format permutation operators.
- __Fully Traced Model__: `stable-fast` improves the `torch.jit.trace` interface to make it more proper for tracing complex models. Nearly every part of `StableDiffusionPipeline` can be traced and converted to __TorchScript__. It is more stable than `torch.compile` and has a significantly lower CPU overhead than `torch.compile` and supports __ControlNet__ and __LoRA__.
- __CUDA Graph__: `stable-fast` can capture the UNet structure into CUDA Graph format, which can reduce the CPU overhead when the batch size is small.
- __Fused Multihead Attention__: `stable-fast` just uses xformers and make it compatible with __TorchScript__.

### Performance Comparation

#### A100 SXM 80GB (SD v1.5, 512 * 512)

| Framework       | Performance |
| --------------- | ----------- |
| __Stable Fast__ | __60 it/s__ |
| Vanilla PyTorch | 23 it/s     |
| AITemplate      | 44 it/s     |
| TensorRT        | 52 it/s     |
| OneFlow         | 55 it/s     |

## Usage

### Installation

__NOTE: `stable-fast` is currently only tested on Linux. You need to install PyTorch with CUDA support at first (version 1.12 - 2.1 is suggested).__

```bash
# Install PyTorch with CUDA first
# pip3 install torch==x.x.x+cuxxx

# Clone this repository.
git clone https://github.com/chengzeyi/stable-fast.git

# Build wheel package
SFAST_VERSION_SUFFIX=+torchxxx.cuxxx python3 setup.py bdist_wheel

# Or just install it
pip3 install '.[diffusers,xformers,triton]'
```

__NOTE: Any usage outside `sfast.compilers` is not guaranteed to be backward compatible.__
__NOTE: To get the best performance, `xformers` and OpenAI's `triton` need to be installed and enabled__.

### Some Common Methods To Speed Up PyTorch

```bash
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
import logging
import torch
from diffusers import StableDiffusionPipeline
from sfast.compilers.stable_diffusion_pipeline_compiler import (compile,
                                                                CompilationConfig
                                                                )

logger = logging.getLogger()

def load_model():
    model = StableDiffusionPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', torch_dtype=torch.float16)
    model.safety_checker = None
    model.to(torch.device('cuda'))
    return model

model = load_model()

config = CompilationConfig.Default()
try:
    import xformers
    config.enable_xformers = True
except ImportError:
    logger.warning('xformers not installed, skip')
try:
    import triton
    config.enable_triton = True
except ImportError:
    logger.warning('triton not installed, skip')
config.enable_cuda_graph = True
compiled_model = compile(model, config)

kwarg_inputs = dict(
    prompt=
    '(masterpiece:1,2), best quality, masterpiece,best detail face, lineart, monochrome, a sexy girl',
    height=512,
    width=512,
    num_inference_steps=50,
    num_images_per_prompt=1,
)
output_image = compiled_model(**kwarg_inputs).images[0]
```
