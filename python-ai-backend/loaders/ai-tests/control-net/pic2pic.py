import torch
import os
from huggingface_hub import HfApi
from pathlib import Path
from diffusers.utils import load_image
from PIL import Image
import numpy as np

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)

checkpoint = "lllyasviel/control_v11e_sd15_ip2p"

image = Image.open('./snapshots/image.png').convert("RGB").resize((512, 768))

prompt = "turn her in to an anime girl, vivid, high detail, studio gibli"

controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V1.4", controlnet=controlnet, torch_dtype=torch.float16
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")
pipe.enable_attention_slicing()
generator = torch.manual_seed(0)
image = pipe(prompt, negative_prompt='lowres, poorly drawn face, out of frame, poorly drawn hands, too many fingers, blurry, bad art, blurred, disfigured, deformed, closed eyes', num_inference_steps=50, generator=generator, image=image).images[0]
#torch.cuda.empty_cache()
image.save('./snapshots/image_out_clothing.png')
