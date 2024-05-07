REPO = None
FACE_ANALYSIS_ROOT = None
MODEL = 'wangqixun/YamerMIX_v8'
VARIANT = None
CUSTOM_PIPELINE = None
SCHEDULER = 'EulerAncestralDiscreteScheduler'
LORA = None
CONTROLNET = 'InstantX/InstantID'
STEPS = 30
PROMPT = 'film noir style, ink sketch|vector, male man, highly detailed, sharp focus, ultra sharpness, monochrome, high contrast, dramatic shadows, 1940s style, mysterious, cinematic'
NEGATIVE_PROMPT = 'ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, vibrant, colorful'
SEED = None
WARMUPS = 3
BATCH = 1
HEIGHT = None
WIDTH = None
INPUT_IMAGE = 'https://github.com/InstantID/InstantID/blob/main/examples/musk_resize.jpeg?raw=true'
CONTROL_IMAGE = None
OUTPUT_IMAGE = None
EXTRA_CALL_KWARGS = '''{
    "controlnet_conditioning_scale": 0.8,
    "ip_adapter_scale": 0.8
}'''

import sys
import os
import importlib
import inspect
import argparse
import time
import json
import torch
from PIL import (Image, ImageDraw)
import numpy as np
import cv2
from huggingface_hub import snapshot_download
from diffusers.utils import load_image
from insightface.app import FaceAnalysis
from sfast.compilers.diffusion_pipeline_compiler import (compile,
                                                         CompilationConfig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=str, default=REPO)
    parser.add_argument('--face-analysis-root',
                        type=str,
                        default=FACE_ANALYSIS_ROOT)
    parser.add_argument('--model', type=str, default=MODEL)
    parser.add_argument('--variant', type=str, default=VARIANT)
    parser.add_argument('--custom-pipeline', type=str, default=CUSTOM_PIPELINE)
    parser.add_argument('--scheduler', type=str, default=SCHEDULER)
    parser.add_argument('--lora', type=str, default=LORA)
    parser.add_argument('--controlnet', type=str, default=CONTROLNET)
    parser.add_argument('--steps', type=int, default=STEPS)
    parser.add_argument('--prompt', type=str, default=PROMPT)
    parser.add_argument('--negative-prompt', type=str, default=NEGATIVE_PROMPT)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--warmups', type=int, default=WARMUPS)
    parser.add_argument('--batch', type=int, default=BATCH)
    parser.add_argument('--height', type=int, default=HEIGHT)
    parser.add_argument('--width', type=int, default=WIDTH)
    parser.add_argument('--extra-call-kwargs',
                        type=str,
                        default=EXTRA_CALL_KWARGS)
    parser.add_argument('--input-image', type=str, default=INPUT_IMAGE)
    parser.add_argument('--control-image', type=str, default=CONTROL_IMAGE)
    parser.add_argument('--output-image', type=str, default=OUTPUT_IMAGE)
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
    # My implementation can handle dynamic shape with increased need for GPU memory.
    # But when your GPU VRAM is insufficient or the image resolution is high,
    # CUDA Graph could cause less efficient VRAM utilization and slow down the inference,
    # especially when on Windows or WSL which has the "shared VRAM" mechanism.
    # If you meet problems related to it, you should disable it.
    config.enable_cuda_graph = True

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

    assert args.repo is not None, 'Please set `--repo` to the local path of the cloned repo of https://github.com/InstantID/InstantID'
    assert args.controlnet is not None, 'Please set `--controlnet` to the name or path of the controlnet'
    assert args.face_analysis_root is not None, 'Please set `--face-analysis-root` to the path of the working directory of insightface.app.FaceAnalysis'
    assert os.path.isdir(
        os.path.join(args.face_analysis_root, 'models', 'antelopev2')
    ), f'Please download models from https://drive.google.com/file/d/18wEUfMNohBJ4K3Ly5wpTejPfDzp-8fI8/view?usp=sharing and extract to {os.path.join(args.face_analysis_root, "models", "antelopev2")}'
    assert args.input_image is not None, 'Please set `--input-image` to the path of the input image'

    sys.path.insert(0, args.repo)
    from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps

    if os.path.exists(args.controlnet):
        controlnet = args.controlnet
    else:
        controlnet = snapshot_download(args.controlnet)

    app = FaceAnalysis(
        name='antelopev2',
        root=args.face_analysis_root,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    face_adapter = os.path.join(controlnet, 'ip-adapter.bin')
    controlnet_path = os.path.join(controlnet, 'ControlNetModel')

    model = load_model(
        StableDiffusionXLInstantIDPipeline,
        args.model,
        variant=args.variant,
        custom_pipeline=args.custom_pipeline,
        scheduler=args.scheduler,
        lora=args.lora,
        controlnet=controlnet_path,
    )

    model.load_ip_adapter_instantid(face_adapter)

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
    input_image = input_image.resize((width, height), Image.LANCZOS)

    face_image = input_image
    face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
    face_info = sorted(
        face_info,
        key=lambda x:
        (x['bbox'][2] - x['bbox'][0]) * x['bbox'][3] - x['bbox'][1])[
            -1]  # only use the maximum face
    face_emb = face_info['embedding']
    face_kps = draw_kps(face_image, face_info['kps'])

    def get_kwarg_inputs():
        kwarg_inputs = dict(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            image_embeds=face_emb,
            image=face_kps,
            height=height,
            width=width,
            num_inference_steps=args.steps,
            num_images_per_prompt=args.batch,
            generator=None if args.seed is None else torch.Generator(
                device="cuda").manual_seed(args.seed),
            **(dict() if args.extra_call_kwargs is None else json.loads(
                args.extra_call_kwargs)),
        )
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
    output_images = model(**kwarg_inputs).images
    end = time.time()

    # Let's view it in terminal!
    from sfast.utils.term_image import print_image

    for image in output_images:
        print_image(image, max_width=80)

    print(f'Inference time: {end - begin:.3f}s')
    iter_per_sec = iter_profiler.get_iter_per_sec()
    if iter_per_sec is not None:
        print(f'Iterations per second: {iter_per_sec:.3f}')
    peak_mem = torch.cuda.max_memory_allocated()
    print(f'Peak memory: {peak_mem / 1024**3:.3f}GiB')

    if args.output_image is not None:
        output_images[0].save(args.output_image)


if __name__ == '__main__':
    main()
