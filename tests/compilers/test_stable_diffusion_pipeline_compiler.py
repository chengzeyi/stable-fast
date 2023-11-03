import pytest

import logging
import functools
import os
import glob
import cv2
import PIL
import torch
from sfast.compilers.stable_diffusion_pipeline_compiler import (
    compile, CompilationConfig)
from sfast.profile.auto_profiler import AutoProfiler
from sfast.utils.term_image import display_image
from sfast.utils.compute_precision import low_compute_precision

logger = logging.getLogger()


def get_images_from_path(path):
    image_paths = sorted(glob.glob(os.path.join(path, '*.*')))
    images = [
        cv2.imread(image_path, cv2.IMREAD_COLOR) for image_path in image_paths
    ]
    images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]
    return images


def test_benchmark_sd15_model(sd15_model_path):
    benchmark_sd_model(
        sd15_model_path,
        kwarg_inputs=lambda: dict(
            prompt=
            '(masterpiece:1,2), best quality, masterpiece, best detail face, realistic, unreal engine, a sexy girl',
            height=512,
            width=512,
            num_inference_steps=30,
        ))


def test_benchmark_sd21_model(sd21_model_path):
    benchmark_sd_model(
        sd21_model_path,
        kwarg_inputs=lambda: dict(
            prompt=
            '(masterpiece:1,2), best quality, masterpiece, best detail face, Van Gogh style, a beautiful girl',
            height=512,
            width=512,
            num_inference_steps=30,
        ))


def test_benchmark_sdxl_model(sdxl_model_path):
    from diffusers import StableDiffusionXLPipeline

    benchmark_sd_model(
        sdxl_model_path,
        kwarg_inputs=lambda: dict(
            prompt=
            '(masterpiece:1,2), best quality, masterpiece, best detail face, romantic style, a beautiful girl',
            height=512,
            width=512,
            num_inference_steps=30,
        ),
        model_class=StableDiffusionXLPipeline)


def test_benchmark_sd15_model_with_controlnet(sd15_model_path,
                                              sd_controlnet_canny_model_path,
                                              diffusers_dog_example_path):
    from diffusers import StableDiffusionControlNetPipeline

    dog_image = get_images_from_path(diffusers_dog_example_path)[0]
    dog_image = cv2.resize(dog_image, (512, 512))
    dog_image_canny = cv2.Canny(dog_image, 100, 200)
    dog_image_canny = PIL.Image.fromarray(dog_image_canny)

    benchmark_sd_model(
        sd15_model_path,
        kwarg_inputs=lambda: dict(
            prompt=
            '(masterpiece:1,2), best quality, masterpiece, best detail face, Van Gogh style, a beautiful girl',
            height=512,
            width=512,
            num_inference_steps=30,
            image=dog_image_canny,
        ),
        model_class=StableDiffusionControlNetPipeline,
        controlnet_model_path=sd_controlnet_canny_model_path,
    )


def call_model(model, inputs=None, kwarg_inputs=None):
    inputs = tuple() if inputs is None else inputs()
    kwarg_inputs = dict() if kwarg_inputs is None else kwarg_inputs()
    torch.manual_seed(0)
    output_image = model(*inputs, **kwarg_inputs).images[0]
    return output_image


def benchmark_sd_model(model_path,
                       kwarg_inputs,
                       model_class=None,
                       scheduler_class=None,
                       controlnet_model_path=None):
    from diffusers import (
        StableDiffusionPipeline,
        EulerAncestralDiscreteScheduler,
    )

    if model_class is None:
        model_class = StableDiffusionPipeline
    if scheduler_class is None:
        scheduler_class = EulerAncestralDiscreteScheduler

    def load_model():
        model_init_kwargs = {}

        if controlnet_model_path is not None:
            from diffusers import ControlNetModel

            controlnet_model = ControlNetModel.from_pretrained(
                controlnet_model_path, torch_dtype=torch.float16)
            model_init_kwargs['controlnet'] = controlnet_model

        model = model_class.from_pretrained(model_path,
                                            torch_dtype=torch.float16,
                                            **model_init_kwargs)
        if scheduler_class is not None:
            model.scheduler = scheduler_class.from_config(
                model.scheduler.config)
        model.safety_checker = None
        model.to(torch.device('cuda'))
        return model

    call_model_ = functools.partial(call_model, kwarg_inputs=kwarg_inputs)

    with AutoProfiler(.02) as profiler, low_compute_precision():
        logger.info('Benchmarking StableDiffusionPipeline')
        model = load_model()

        def call_original_model():
            return call_model_(model)

        for _ in range(3):
            call_original_model()

        output_image = profiler.with_cProfile(call_original_model)()
        display_image(output_image)

        if hasattr(torch, 'compile'):
            logger.info(
                'Benchmarking StableDiffusionPipeline with torch.compile')
            model.unet.to(memory_format=torch.channels_last)
            model.unet = torch.compile(model.unet)
            if hasattr(model, 'controlnet'):
                model.controlnet.to(memory_format=torch.channels_last)
                model.controlnet = torch.compile(model.controlnet)

            def call_torch_compiled_model():
                return call_model_(model)

            for _ in range(3):
                call_torch_compiled_model()

            output_image = profiler.with_cProfile(call_torch_compiled_model)()
            display_image(output_image)

        del model

        logger.info('Benchmarking compiled StableDiffusionPipeline')
        config = CompilationConfig.Default()
        compiled_model = compile(load_model(), config)

        def call_compiled_model():
            return call_model_(compiled_model)

        for _ in range(3):
            call_compiled_model()

        output_image = profiler.with_cProfile(call_compiled_model)()
        display_image(output_image)

        del compiled_model

        logger.info(
            'Benchmarking compiled StableDiffusionPipeline with xformers, Triton and CUDA Graph'
        )
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
        compiled_model = compile(load_model(), config)

        def call_faster_compiled_model():
            return call_model_(compiled_model)

        for _ in range(3):
            call_faster_compiled_model()

        output_image = profiler.with_cProfile(call_faster_compiled_model)()
        display_image(output_image)

        del compiled_model
