import torch
from diffusers import LCMScheduler, AutoPipelineForText2Image, DiffusionPipeline
from sfast.compilers.stable_diffusion_pipeline_compiler import (
    compile, CompilationConfig)
import numpy as np
from PIL import Image

base_model_path = "../stable-diffusion-v1-5"
lcm_path = "latent-consistency/lcm-lora-sdv1-5"


def load_model():
    model = DiffusionPipeline.from_pretrained(base_model_path,
                                              torch_dtype=torch.float16,
                                              safety_checker=None,
                                              use_safetensors=True)

    model.scheduler = LCMScheduler.from_config(model.scheduler.config)
    model.safety_checker = None
    model.to(torch.device('cuda'))
    # model.unet.load_attn_procs(lcm_path)
    model.load_lora_weights(lcm_path)
    model.fuse_lora()
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
    prompt = "a rendering of a living room with a couch and a tv"
    negative_prompt = "ugly,logo,pixelated,lowres,text,word,cropped,low quality,normal quality,username,watermark,signature,blurry,soft,NSFW,painting,cartoon,hang,occluded objects,Fisheye View"

    model = load_model()
    model = compile_model(model)

    kwarg_inputs = dict(
        prompt=prompt,
        # negative_prompt=negative_prompt,
        width=768,
        height=512,
        num_inference_steps=7,
        num_images_per_prompt=1,
        guidance_scale=0.0,
    )

    # NOTE: Warm it up.
    # The initial calls will trigger compilation and might be very slow.
    # After that, it should be very fast.
    for _ in range(3):
        output_image = model(**kwarg_inputs).images[0]

    # Let's see it!
    # Note: Progress bar might work incorrectly due to the async nature of CUDA.

    img_total = []
    import time
    begin = time.time()
    for i in range(2):
        output_image = model(
            **kwarg_inputs,
        ).images

        img_row = []
        for img in output_image:
            img_row.append(np.asarray(img))
        img = np.hstack(img_row)
        img_total.append(img)
    print(time.time() - begin)
    image = np.vstack(img_total)
    # cv2.putText(image,prompt,(40,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),3)

    image = Image.fromarray(image)
    from sfast.utils.term_image import print_image
    print_image(image, max_width=80)


if __name__ == '__main__':
    main()
