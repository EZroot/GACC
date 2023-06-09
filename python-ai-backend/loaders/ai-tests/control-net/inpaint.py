# !pip install transformers accelerate
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDIMScheduler
from diffusers.utils import load_image
import numpy as np
import torch
from PIL import Image

init_image = Image.open('./snapshots/image.png').convert("RGB").resize((512, 768))

generator = torch.Generator(device="cpu").manual_seed(1)

mask_image = Image.open('./snapshots/masked_image.png').convert("RGB").resize((512, 768))


def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

    assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image


control_image = make_inpaint_condition(init_image, mask_image)

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16
)
pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V1.4", controlnet=controlnet, torch_dtype=torch.float16
)

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

# generate image
image = pipe(
    "a fist, 8k, vivid, high detail",
    negative_prompt='lowres, poorly drawn face, out of frame, poorly drawn hands, too many fingers, blurry, bad art, blurred, disfigured, deformed, closed eyes',
    num_inference_steps=50,
    generator=generator,
    image=init_image,
    mask_image=mask_image,
    control_image=control_image,
).images[0]

image.save('./gen_pics/newpic_inpaint.png')