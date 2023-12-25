import torch
from diffusers import AutoPipelineForText2Image, EulerDiscreteScheduler, ControlNetModel
from diffusers.utils import load_image
from sfast.compilers.diffusion_pipeline_compiler import (compile,
                                                         CompilationConfig)
import numpy as np
import cv2
from PIL import Image

CUDA_DEVICE = "cuda:0"


def canny_process(image, width, height):
    np_image = cv2.resize(image, (width, height))
    np_image = cv2.Canny(np_image, 100, 200)
    np_image = np_image[:, :, None]
    np_image = np.concatenate([np_image, np_image, np_image], axis=2)
    # canny_image = Image.fromarray(np_image)
    return Image.fromarray(np_image)


def reference_process(image, width, height):
    np_image = cv2.resize(image, (width, height))
    return Image.fromarray(np_image)


def load_model():
    extra_kwargs = {}
    # extra_kwargs['variant'] = variant

    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_canny",
        torch_dtype=torch.float16,
        variant="fp16",
        name="diffusion_pytorch_model.fp16.safetensors",
        use_safetensors=True)
    extra_kwargs['controlnet'] = controlnet
    model = AutoPipelineForText2Image.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        **extra_kwargs)
    model.scheduler = EulerDiscreteScheduler.from_config(
        model.scheduler.config)
    model.safety_checker = None
    model.load_ip_adapter("h94/IP-Adapter",
                          subfolder="models",
                          weight_name="ip-adapter_sd15.safetensors")
    model.to(torch.device(CUDA_DEVICE))

    return model


def compile_model(model):
    config = CompilationConfig.Default()
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
    config.enable_cuda_graph = True

    model = compile(model, config)

    return model


if __name__ == "__main__":
    control_img = 'https://huggingface.co/lllyasviel/control_v11p_sd15_canny/resolve/main/images/bird.png'
    reference_img = 'https://huggingface.co/datasets/diffusers/dog-example/resolve/main/alvan-nee-eoqnr8ikwFE-unsplash.jpeg'

    width = 768
    height = 512

    control_img = load_image(control_img)
    reference_img = load_image(reference_img)
    control_img = np.array(control_img)
    reference_img = np.array(reference_img)
    control_img = canny_process(control_img, width, height)
    reference_img = reference_process(reference_img, width, height)

    model = load_model()
    model = compile_model(model)
    seed = -1
    batch_size = 4
    generator = torch.Generator(device=CUDA_DEVICE).manual_seed(seed)
    prompt = "dog"
    negative_prompt = ""
    num_inference_steps = 20
    guidance_scale = 7.5
    controlnet_conditioning_scale = 1.0

    for _ in range(3):
        images = model(
            prompt=[prompt] * batch_size,
            negative_prompt=[negative_prompt] * batch_size,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=1,
            guidance_scale=guidance_scale,
            ip_adapter_image=[reference_img] * batch_size,
            image=[control_img] * batch_size,
            generator=generator,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
        ).images

    from sfast.utils.term_image import print_image

    for image in images:
        print_image(image, max_width=80)
