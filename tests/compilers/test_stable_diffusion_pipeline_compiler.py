import pytest

import logging
import functools
import packaging.version
import os
import gc
import glob
import copy
import cv2
import PIL
import torch
from sfast.compilers.diffusion_pipeline_compiler import (
    compile,
    CompilationConfig,
)
from sfast.profile.auto_profiler import AutoProfiler
from sfast.utils.term_image import print_image
from sfast.utils.compute_precision import low_compute_precision
from sfast.utils.patch import patch_module

logger = logging.getLogger()

basic_kwarg_inputs = dict(
    prompt=
    '(masterpiece:1,2), best quality, masterpiece, best detailed face, realistic, unreal engine, a beautiful girl',
    height=512,
    width=512,
    num_inference_steps=30,
)

basic_img2img_kwarg_inputs = dict(
    prompt='cartonized',
    num_inference_steps=30,
)


def display_image(image):
    print_image(image, max_width=80)


def get_images_from_path(path):
    image_paths = sorted(glob.glob(os.path.join(path, '*.*')))
    images = [
        cv2.imread(image_path, cv2.IMREAD_COLOR) for image_path in image_paths
    ]
    images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]
    return images


def test_benchmark_sd15_model(sd15_model_path, skip_comparsion=False):
    benchmark_sd_model(
        sd15_model_path,
        kwarg_inputs=basic_kwarg_inputs,
        skip_comparsion=skip_comparsion,
    )


def test_compile_sd15_model(sd15_model_path, skip_comparsion=True):
    test_benchmark_sd15_model(sd15_model_path, skip_comparsion=skip_comparsion)


def test_benchmark_sd15_model_with_tiny_vae(sd15_model_path,
                                            skip_comparsion=False):
    benchmark_sd_model(
        sd15_model_path,
        kwarg_inputs=basic_kwarg_inputs,
        skip_comparsion=skip_comparsion,
        enable_tiny_vae=True,
    )


def test_compile_sd15_model_with_tiny_vae(sd15_model_path,
                                          skip_comparsion=True):
    test_benchmark_sd15_model_with_tiny_vae(sd15_model_path,
                                            skip_comparsion=skip_comparsion)


def test_benchmark_quantized_sd15_model(sd15_model_path,
                                        skip_comparsion=False):
    benchmark_sd_model(
        sd15_model_path,
        kwarg_inputs=basic_kwarg_inputs,
        skip_comparsion=skip_comparsion,
        quantize=True,
    )


def test_compile_quantized_sd15_model(sd15_model_path, skip_comparsion=True):
    test_benchmark_quantized_sd15_model(sd15_model_path,
                                        skip_comparsion=skip_comparsion)


def test_benchmark_sd15_model_with_lora(sd15_model_path,
                                        sd15_lora_t4_path,
                                        sd15_lora_dog_path,
                                        skip_comparsion=False):
    benchmark_sd_model(
        sd15_model_path,
        kwarg_inputs=basic_kwarg_inputs,
        lora_a_path=sd15_lora_t4_path,
        lora_b_path=sd15_lora_dog_path,
        skip_comparsion=skip_comparsion,
    )


def test_compile_sd15_model_with_lora(sd15_model_path,
                                      sd15_lora_t4_path,
                                      sd15_lora_dog_path,
                                      skip_comparsion=True):
    benchmark_sd_model(
        sd15_model_path,
        kwarg_inputs=basic_kwarg_inputs,
        lora_a_path=sd15_lora_t4_path,
        lora_b_path=sd15_lora_dog_path,
        skip_comparsion=skip_comparsion,
    )


def test_benchmark_sd15_model_with_controlnet(sd15_model_path,
                                              sd_controlnet_canny_model_path,
                                              diffusers_dog_example_path,
                                              skip_comparsion=False):
    from diffusers import StableDiffusionControlNetPipeline

    dog_image = get_images_from_path(diffusers_dog_example_path)[0]
    dog_image = cv2.resize(dog_image, (512, 512))
    dog_image_canny = cv2.Canny(dog_image, 100, 200)
    dog_image_canny = PIL.Image.fromarray(dog_image_canny)

    benchmark_sd_model(
        sd15_model_path,
        kwarg_inputs=dict(
            **basic_kwarg_inputs,
            image=dog_image_canny,
        ),
        model_class=StableDiffusionControlNetPipeline,
        controlnet_model_path=sd_controlnet_canny_model_path,
        skip_comparsion=skip_comparsion,
    )


def test_compile_sd15_model_with_controlnet(sd15_model_path,
                                            sd_controlnet_canny_model_path,
                                            diffusers_dog_example_path,
                                            skip_comparsion=True):
    test_benchmark_sd15_model_with_controlnet(sd15_model_path,
                                              sd_controlnet_canny_model_path,
                                              diffusers_dog_example_path,
                                              skip_comparsion=skip_comparsion)


def test_benchmark_sd15_model_with_img2img(sd15_model_path,
                                           diffusers_dog_example_path,
                                           skip_comparsion=False):
    from diffusers import StableDiffusionImg2ImgPipeline

    dog_image = get_images_from_path(diffusers_dog_example_path)[0]
    dog_image = cv2.resize(dog_image, (512, 512))
    dog_image = PIL.Image.fromarray(dog_image)

    benchmark_sd_model(
        sd15_model_path,
        kwarg_inputs=dict(
            **basic_img2img_kwarg_inputs,
            image=dog_image,
        ),
        model_class=StableDiffusionImg2ImgPipeline,
        skip_comparsion=skip_comparsion,
    )


def test_compile_sd15_model_with_img2img(sd15_model_path,
                                         diffusers_dog_example_path,
                                         skip_comparsion=True):
    test_benchmark_sd15_model_with_img2img(sd15_model_path,
                                           diffusers_dog_example_path,
                                           skip_comparsion=skip_comparsion)


def test_benchmark_sd21_model(sd21_model_path, skip_comparsion=False):
    kwarg_inputs = copy.deepcopy(basic_kwarg_inputs)
    kwarg_inputs['height'] = 768
    kwarg_inputs['width'] = 768

    benchmark_sd_model(
        sd21_model_path,
        kwarg_inputs=kwarg_inputs,
        skip_comparsion=skip_comparsion,
    )


def test_compile_sd21_model(sd21_model_path, skip_comparsion=True):
    test_benchmark_sd21_model(sd21_model_path, skip_comparsion=skip_comparsion)


def test_benchmark_sdxl_model(sdxl_model_path, skip_comparsion=False):
    from diffusers import StableDiffusionXLPipeline

    kwarg_inputs = copy.deepcopy(basic_kwarg_inputs)
    kwarg_inputs['height'] = 768
    kwarg_inputs['width'] = 768

    benchmark_sd_model(
        sdxl_model_path,
        kwarg_inputs=kwarg_inputs,
        model_class=StableDiffusionXLPipeline,
        skip_comparsion=skip_comparsion,
    )


def test_compile_sdxl_model(sdxl_model_path, skip_comparsion=True):
    test_benchmark_sdxl_model(sdxl_model_path, skip_comparsion=skip_comparsion)


def test_benchmark_quantized_sdxl_model(sdxl_model_path,
                                        skip_comparsion=False):
    from diffusers import StableDiffusionXLPipeline

    kwarg_inputs = copy.deepcopy(basic_kwarg_inputs)
    kwarg_inputs['height'] = 768
    kwarg_inputs['width'] = 768

    benchmark_sd_model(
        sdxl_model_path,
        kwarg_inputs=kwarg_inputs,
        model_class=StableDiffusionXLPipeline,
        skip_comparsion=skip_comparsion,
        quantize=True,
    )


def test_compile_quantized_sdxl_model(sdxl_model_path, skip_comparsion=True):
    test_benchmark_quantized_sdxl_model(sdxl_model_path,
                                        skip_comparsion=skip_comparsion)


def call_model(model, inputs=None, kwarg_inputs=None):
    inputs = tuple() if inputs is None else inputs() if callable(
        inputs) else inputs
    kwarg_inputs = dict() if kwarg_inputs is None else kwarg_inputs(
    ) if callable(kwarg_inputs) else kwarg_inputs
    torch.manual_seed(0)
    output_image = model(*inputs, **kwarg_inputs).images[0]
    return output_image


def benchmark_sd_model(
    model_path,
    kwarg_inputs,
    model_class=None,
    scheduler_class=None,
    controlnet_model_path=None,
    enable_cuda_graph=True,
    skip_comparsion=False,
    lora_a_path=None,
    lora_b_path=None,
    quantize=False,
    enable_tiny_vae=False,
):
    from diffusers import (
        StableDiffusionPipeline,
        EulerAncestralDiscreteScheduler,
    )

    if model_class is None:
        model_class = StableDiffusionPipeline
    if scheduler_class is None:
        scheduler_class = EulerAncestralDiscreteScheduler

    def load_model():
        with torch.no_grad():
            model_init_kwargs = {}

            if controlnet_model_path is not None:
                from diffusers import ControlNetModel

                controlnet_model = ControlNetModel.from_pretrained(
                    controlnet_model_path, torch_dtype=torch.float16)
                model_init_kwargs['controlnet'] = controlnet_model

            gc.collect()
            torch.cuda.empty_cache()
            before_memory = torch.cuda.memory_allocated()

            model = model_class.from_pretrained(model_path,
                                                torch_dtype=torch.float16,
                                                **model_init_kwargs)
            if scheduler_class is not None:
                model.scheduler = scheduler_class.from_config(
                    model.scheduler.config)

            if enable_tiny_vae:
                from diffusers import AutoencoderTiny
                model.vae = AutoencoderTiny.from_pretrained(
                    'madebyollin/taesd', torch_dtype=torch.float16)

            model.safety_checker = None
            model.to(torch.device('cuda'))

            gc.collect()
            after_memory = torch.cuda.memory_allocated()
            logger.info(
                f'Loaded model with {after_memory - before_memory} bytes allocated'
            )
            if quantize:

                def quantize_unet(m):
                    from diffusers.utils import USE_PEFT_BACKEND
                    assert USE_PEFT_BACKEND
                    m = torch.quantization.quantize_dynamic(m,
                                                            {torch.nn.Linear},
                                                            dtype=torch.qint8,
                                                            inplace=True)
                    return m

                model.unet = quantize_unet(model.unet)
                if hasattr(model, 'controlnet'):
                    model.controlnet = quantize_unet(model.controlnet)

                gc.collect()
                after_memory = torch.cuda.memory_allocated()
                logger.info(
                    f'Quantized model with {after_memory - before_memory} bytes allocated'
                )

            if lora_a_path is not None:
                model.unet.load_attn_procs(lora_a_path)

            # This is only for benchmarking purpose.
            # Patch the scheduler to force a synchronize to make the progress bar work
            # (But not accurate).
            '''
            scheduler_step = model.scheduler.step

            def scheduler_step_(*args, **kwargs):
                ret = scheduler_step(*args, **kwargs)
                torch.cuda.synchronize()
                return ret

            model.scheduler.step = scheduler_step_

            # Also patch the image processor.
            image_processor_postprocess = model.image_processor.postprocess

            def image_processor_postprocess_(*args, **kwargs):
                torch.cuda.synchronize()
                ret = image_processor_postprocess(*args, **kwargs)
                return ret

            model.image_processor.postprocess = image_processor_postprocess_
            '''

            return model

    call_model_ = functools.partial(call_model, kwarg_inputs=kwarg_inputs)

    with AutoProfiler(0.02) as profiler, low_compute_precision():
        if not skip_comparsion:
            logger.info('Benchmarking StableDiffusionPipeline')
            model = load_model()

            def call_original_model():
                return call_model_(model)

            for _ in range(3):
                call_original_model()

            output_image = profiler.with_cProfile(call_original_model)()
            display_image(output_image)

            del model

            if hasattr(torch, 'compile') and not quantize:
                model = load_model()
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

                output_image = profiler.with_cProfile(
                    call_torch_compiled_model)()
                display_image(output_image)

                del model

            # logger.info('Benchmarking compiled StableDiffusionPipeline')
            # config = CompilationConfig.Default()
            # compiled_model = compile(load_model(), config)

            # def call_compiled_model():
            #     return call_model_(compiled_model)

            # for _ in range(3):
            #     call_compiled_model()

            # output_image = profiler.with_cProfile(call_compiled_model)()
            # display_image(output_image)

            # del compiled_model

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
        # config.trace_scheduler = True
        config.enable_cuda_graph = enable_cuda_graph
        compiled_model = compile(load_model(), config)

        def call_faster_compiled_model():
            return call_model_(compiled_model)

        for _ in range(3):
            call_faster_compiled_model()

        output_image = profiler.with_cProfile(call_faster_compiled_model)()
        display_image(output_image)

        if lora_a_path is not None and lora_b_path is not None and packaging.version.parse(
                torch.__version__) >= packaging.version.parse('2.1.0'):
            # load_state_dict with assign=True requires torch >= 2.1.0

            def update_state_dict(dst, src):
                for key, value in src.items():
                    # Do inplace copy.
                    # As the traced forward function shares the same reference of the tensors,
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

            output_image = profiler.with_cProfile(call_faster_compiled_model)()
            display_image(output_image)

        del compiled_model
