from io import BytesIO

import torch
from diffusers import DiffusionPipeline, UNet2DConditionModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from safetensors.torch import load_file as load_safetensors
from torch import autocast

# Define model ID and device
model_id = "DGSpitzer/Cyberpunk-Anime-Diffusion"
device = "cuda"

# Prompts
prompt = "portrait of a girl in dgs illustration style, Anime girl, female soldier working in a cyberpunk city, cleavage, ((perfect femine face)), intricate, 8k, highly detailed, shy, digital painting, intense, sharp focus"

negative_prompt = "EasyNegative, extra fingers,fewer fingers,"

# Parameters
steps = 20
sampler = "DPM++ 2M Karras"
cfg_scale = 10
size = (448, 768)
denoising_strength = 0.6
hires_upscale = 1.8
hires_upscaler = "Latent"

# Load pipeline
pipeline = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(
    device
)


# Generate the image
with autocast(device):
    image = pipeline(
        prompt,
        guidance_scale=8.5,
    ).images[0]

# Save and display the image
image.save("image.png")
buffer = BytesIO()
image.save(buffer, format="PNG")
