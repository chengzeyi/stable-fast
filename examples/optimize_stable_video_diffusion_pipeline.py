MODEL = 'stabilityai/stable-video-diffusion-img2vid-xt'
VARIANT = None
CUSTOM_PIPELINE = None
SCHEDULER = None
LORA = None
CONTROLNET = None
STEPS = 25
SEED = None
WARMUPS = 1
FRAMES = None
BATCH = 1
HEIGHT = 576
WIDTH = 1024
FPS = 7
DECODE_CHUNK_SIZE = 4
INPUT_IMAGE = 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png?download=true'
CONTROL_IMAGE = None
OUTPUT_VIDEO = None
EXTRA_CALL_KWARGS = None

import importlib
import inspect
import argparse
import time
import json
import torch
from PIL import (Image, ImageDraw)
from diffusers.utils import load_image, export_to_video
from sfast.compilers.diffusion_pipeline_compiler import (compile,
                                                         CompilationConfig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=MODEL)
    parser.add_argument('--variant', type=str, default=VARIANT)
    parser.add_argument('--custom-pipeline', type=str, default=CUSTOM_PIPELINE)
    parser.add_argument('--scheduler', type=str, default=SCHEDULER)
    parser.add_argument('--lora', type=str, default=LORA)
    parser.add_argument('--controlnet', type=str, default=CONTROLNET)
    parser.add_argument('--steps', type=int, default=STEPS)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--warmups', type=int, default=WARMUPS)
    parser.add_argument('--frames', type=int, default=FRAMES)
    parser.add_argument('--batch', type=int, default=BATCH)
    parser.add_argument('--height', type=int, default=HEIGHT)
    parser.add_argument('--width', type=int, default=WIDTH)
    parser.add_argument('--fps', type=int, default=FPS)
    parser.add_argument('--decode-chunk-size',
                        type=int,
                        default=DECODE_CHUNK_SIZE)
    parser.add_argument('--extra-call-kwargs',
                        type=str,
                        default=EXTRA_CALL_KWARGS)
    parser.add_argument('--input-image', type=str, default=INPUT_IMAGE)
    parser.add_argument('--control-image', type=str, default=CONTROL_IMAGE)
    parser.add_argument('--output-video', type=str, default=OUTPUT_VIDEO)
    parser.add_argument(
        '--compiler',
        type=str,
        default='sfast',
        choices=['none', 'sfast', 'compile', 'compile-max-autotune'])
    parser.add_argument('--quantize', action='store_true')
    parser.add_argument('--no-fusion', action='store_true')
    return parser.parse_args()


def load_model(pipeline_cls,
               model,
               variant=None,
               custom_pipeline=None,
               scheduler=None,
               lora=None,
               controlnet=None):
    extra_kwargs = {}
    if custom_pipeline is not None:
        extra_kwargs['custom_pipeline'] = custom_pipeline
    if variant is not None:
        extra_kwargs['variant'] = variant
    if controlnet is not None:
        from diffusers import ControlNetModel
        controlnet = ControlNetModel.from_pretrained(controlnet,
                                                     torch_dtype=torch.float16)
        extra_kwargs['controlnet'] = controlnet
    model = pipeline_cls.from_pretrained(model,
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
    # But it can increase the amount of GPU memory used.
    # For StableVideoDiffusionPipeline it is not needed.
    # config.enable_cuda_graph = True

    model = compile(model, config)
    return model


class IterationProfiler:

    def __init__(self):
        self.begin = None
        self.end = None
        self.num_iterations = 0

    def get_iter_per_sec(self):
        if self.begin is None or self.end is None:
            return None
        self.end.synchronize()
        dur = self.begin.elapsed_time(self.end)
        return self.num_iterations / dur * 1000.0

    def callback_on_step_end(self, pipe, i, t, callback_kwargs):
        if self.begin is None:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self.begin = event
        else:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self.end = event
            self.num_iterations += 1
        return callback_kwargs


def main():
    args = parse_args()
    from diffusers import StableVideoDiffusionPipeline

    model = load_model(
        StableVideoDiffusionPipeline,
        args.model,
        variant=args.variant,
        custom_pipeline=args.custom_pipeline,
        scheduler=args.scheduler,
        lora=args.lora,
        controlnet=args.controlnet,
    )

    height = args.height or model.unet.config.sample_size * model.vae_scale_factor
    width = args.width or model.unet.config.sample_size * model.vae_scale_factor

    if args.quantize:

        def quantize_unet(m):
            from diffusers.utils import USE_PEFT_BACKEND
            assert USE_PEFT_BACKEND
            m = torch.quantization.quantize_dynamic(m, {torch.nn.Linear},
                                                    dtype=torch.qint8,
                                                    inplace=True)
            return m

        model.unet = quantize_unet(model.unet)
        if hasattr(model, 'controlnet'):
            model.controlnet = quantize_unet(model.controlnet)

    if args.no_fusion:
        torch.jit.set_fusion_strategy([('STATIC', 0), ('DYNAMIC', 0)])

    if args.compiler == 'none':
        pass
    elif args.compiler == 'sfast':
        model = compile_model(model)
    elif args.compiler in ('compile', 'compile-max-autotune'):
        mode = 'max-autotune' if args.compiler == 'compile-max-autotune' else None
        model.unet = torch.compile(model.unet, mode=mode)
        if hasattr(model, 'controlnet'):
            model.controlnet = torch.compile(model.controlnet, mode=mode)
        model.vae = torch.compile(model.vae, mode=mode)
    else:
        raise ValueError(f'Unknown compiler: {args.compiler}')

    input_image = load_image(args.input_image)
    input_image.resize((width, height), Image.LANCZOS)

    if args.control_image is None:
        if args.controlnet is None:
            control_image = None
        else:
            control_image = Image.new('RGB', (width, height))
            draw = ImageDraw.Draw(control_image)
            draw.ellipse((width // 4, height // 4,
                          width // 4 * 3, height // 4 * 3),
                         fill=(255, 255, 255))
            del draw
    else:
        control_image = load_image(args.control_image)
        control_image = control_image.resize((width, height),
                                             Image.LANCZOS)

    def get_kwarg_inputs():
        kwarg_inputs = dict(
            image=input_image,
            height=height,
            width=width,
            num_inference_steps=args.steps,
            num_videos_per_prompt=args.batch,
            num_frames=args.frames,
            fps=args.fps,
            decode_chunk_size=args.decode_chunk_size,
            generator=None
            if args.seed is None else torch.Generator().manual_seed(args.seed),
            **(dict() if args.extra_call_kwargs is None else json.loads(
                args.extra_call_kwargs)),
        )
        if control_image is not None:
            kwarg_inputs['control_image'] = control_image
        return kwarg_inputs

    # NOTE: Warm it up.
    # The initial calls will trigger compilation and might be very slow.
    # After that, it should be very fast.
    if args.warmups > 0:
        print('Begin warmup')
        for _ in range(args.warmups):
            model(**get_kwarg_inputs())
        print('End warmup')

    # Let's see it!
    # Note: Progress bar might work incorrectly due to the async nature of CUDA.
    kwarg_inputs = get_kwarg_inputs()
    iter_profiler = IterationProfiler()
    if 'callback_on_step_end' in inspect.signature(model).parameters:
        kwarg_inputs[
            'callback_on_step_end'] = iter_profiler.callback_on_step_end
    begin = time.time()
    output_frames = model(**kwarg_inputs).frames
    end = time.time()

    # Let's view it in terminal!
    from sfast.utils.term_image import print_image

    for frames in output_frames:
        concated_image = Image.new('RGB', (width * 2, height))
        concated_image.paste(frames[0], (0, 0))
        concated_image.paste(frames[-1], (args.width, 0))
        print_image(concated_image, max_width=120)

    print(f'Inference time: {end - begin:.3f}s')
    iter_per_sec = iter_profiler.get_iter_per_sec()
    if iter_per_sec is not None:
        print(f'Iterations per second: {iter_per_sec:.3f}')
    peak_mem = torch.cuda.max_memory_allocated()
    print(f'Peak memory: {peak_mem / 1024**3:.3f}GiB')

    if args.output_video is not None:
        export_to_video(output_frames[0], args.output_video, fps=args.fps)


if __name__ == '__main__':
    main()
