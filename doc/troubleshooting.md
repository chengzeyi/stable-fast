# TroubleShooting

## Huge Precision Loss

Try tweaking the config:

```python
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_tf32 = False
```

## Compilation Is SO SLOW. How To Improve It?

Dynamic code generation is usually the cause for slow compilation.
You could disable features related to it to speed up compilation.
But this might slow down your inference.

Disable JIT optimized execution (fusion).
__This can significantly speed up compilation.__

```python
# Wrap your code in this context manager
with torch.jit.optimized_execution(False):
    # Do your things
```

Or disable it globally.

```python
torch.jit.set_fusion_strategy([('STATIC', 0), ('DYNAMIC', 0)])
```

Disable Triton (not suggested).

```python
config.enable_triton = False
```

## Inference Is SO SLOW. What's Wrong?

When your GPU VRAM is insufficient or the image resolution is high,
CUDA Graph could cause less efficient VRAM utilization and slow down the inference.

```python
config.enable_cuda_graph = False
```

## Triton Does Not Work

Triton might be not working properly because it uses cache to store compiled kernels,
especially when you just upgrade `stable-fast` or `triton`.
You could try to clear the cache to fix it.

```bash
rm -rf ~/.triton
```

## Crashes, Invalid Memory Access Or Segmentation Fault

Even in PyTorch's own implementation `torch.compile`, I have encountered crashes and segmentation faults.
It is usually caused by Triton, CUDA Graph or cudaMallocAsync because they are not stable enough.
You could try to remove the `PYTORCH_CUDA_MALLOC_CONF=backend:cudaMallocAsync` environment variable
and disable Triton and CUDA Graph to fix it.

```python
config.enable_triton = False
# or
config.enable_cuda_graph = False
```

## Import Error On Windows

```
ImportError: DLL load failed while importing _C:  The specified module could not be found
```

Make sure you have installed `torch` with CUDA support and your installed version is compatible with your Python and CUDA version.
