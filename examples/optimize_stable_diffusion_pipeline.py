import sys
import time
import torch
from diffusers import (StableDiffusionPipeline,
                       EulerAncestralDiscreteScheduler)
from sfast.compilers.stable_diffusion_pipeline_compiler import (
    compile, CompilationConfig)


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
        sys.argv[1] if len(sys.argv) > 1 else 'runwayml/stable-diffusion-v1-5',
        torch_dtype=torch.float16)

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

model = compile(model, config)

kwarg_inputs = dict(
    prompt=
    '(masterpiece:1,2), best quality, masterpiece, best detail face, a beautiful girl',
    # NOTE: If you use SDXL, you should use a higher resolution to improve the generation quality.
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
