import torch
import torch.nn.functional as F

from diffusers import AutoencoderKL

from sfast.compilers.stable_diffusion_pipeline_compiler import (
    compile_vae,
    CompilationConfig,
)

device = torch.device("cuda:0")

SD_2_1_DIFFUSERS_MODEL = "stabilityai/stable-diffusion-2-1"
variant = {"variant": "fp16"}
vae_orig = AutoencoderKL.from_pretrained(
        SD_2_1_DIFFUSERS_MODEL,
        subfolder="vae",
        torch_dtype=torch.float16,
        **variant,
    )

vae_orig.to(device)

sfast_config = CompilationConfig.Default()
sfast_config.enable_xformers = False
sfast_config.enable_triton = True
sfast_config.enable_cuda_graph = False
vae = compile_vae(vae_orig, sfast_config)

sample_imgs = torch.randn(4, 3, 128, 128, dtype=vae.dtype, device=device)
latents1 = torch.randn(4, 4, 16, 16, dtype=vae.dtype, device=device)

latents = vae.encode(sample_imgs).latent_dist.sample()

sample_imgs_dup = sample_imgs.clone().detach().requires_grad_(True)
latents2 = vae_orig.encode(sample_imgs_dup).latent_dist.sample()
print("Test done")
