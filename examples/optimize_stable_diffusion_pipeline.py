MODEL = 'runwayml/stable-diffusion-v1-5'
VARIANT = None
CUSTOM_PIPELINE = None
SCHEDULER = 'EulerAncestralDiscreteScheduler'
LORA = None
STEPS = 30
PROMPT = 'best quality, realistic, unreal engine, 4K, a beautiful girl'
SEED = None
WARMUPS = 3
BATCH = 1
HEIGHT = 512
WIDTH = 512
EXTRA_CALL_KWARGS = None

from sfast.compilers.stable_diffusion_pipeline_compiler import (
    compile, CompilationConfig)
from diffusers import DiffusionPipeline
import torch
import json
import importlib
import argparse
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=MODEL)
    parser.add_argument('--variant', type=str, default=VARIANT)
    parser.add_argument('--custom-pipeline', type=str, default=CUSTOM_PIPELINE)
    parser.add_argument('--scheduler', type=str, default=SCHEDULER)
    parser.add_argument('--lora', type=str, default=LORA)
    parser.add_argument('--steps', type=int, default=STEPS)
    parser.add_argument('--prompt', type=str, default=PROMPT)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--warmups', type=int, default=WARMUPS)
    parser.add_argument('--batch', type=int, default=BATCH)
    parser.add_argument('--height', type=int, default=HEIGHT)
    parser.add_argument('--width', type=int, default=WIDTH)
    parser.add_argument('--extra-call-kwargs',
                        type=str,
                        default=EXTRA_CALL_KWARGS)
    parser.add_argument('--no-optimize', action='store_true')
    return parser.parse_args()


def load_model(model,
               scheduler=None,
               custom_pipeline=None,
               variant=None,
               lora=None):
    extra_kwargs = {}
    if custom_pipeline is not None:
        extra_kwargs['custom_pipeline'] = custom_pipeline
    if variant is not None:
        extra_kwargs['variant'] = variant
    model = DiffusionPipeline.from_pretrained(model,
                                              torch_dtype=torch.float16,
                                              **extra_kwargs)
    if scheduler is not None:
        scheduler_cls = getattr(importlib.import_module('diffusers'),
                                scheduler)
        model.scheduler = scheduler_cls.from_config(model.scheduler.config)
    if lora is not None:
        model.load_lora_weights(lora)
        model.fuse_lora()
    model.safety_checker = None
    model.to(torch.device('cuda'))
    return model


def compile_model(model):
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
    return model


def main():
    args = parse_args()
    model = load_model(
        args.model,
        scheduler=args.scheduler,
        custom_pipeline=args.custom_pipeline,
        variant=args.variant,
        lora=args.lora,
    )
    if not args.no_optimize:
        model = compile_model(model)

    def get_kwarg_inputs():
        kwarg_inputs = dict(
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            num_images_per_prompt=args.batch,
            generator=None if args.seed is None else torch.Generator(
                device='cuda').manual_seed(args.seed),
            **(dict() if args.extra_call_kwargs is None else json.loads(
                args.extra_call_kwargs)),
        )
        return kwarg_inputs

    # NOTE: Warm it up.
    # The initial calls will trigger compilation and might be very slow.
    # After that, it should be very fast.
    for _ in range(args.warmups):
        model(**get_kwarg_inputs())

    # Let's see it!
    # Note: Progress bar might work incorrectly due to the async nature of CUDA.
    begin = time.time()
    output_images = model(**get_kwarg_inputs()).images
    end = time.time()

    # Let's view it in terminal!
    from sfast.utils.term_image import print_image

    for image in output_images:
        print_image(image, max_width=80)

    print(f'Inference time: {end - begin:.3f}s')


if __name__ == '__main__':
    main()
