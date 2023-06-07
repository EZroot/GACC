import PIL
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float32, safety_checker=None)
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# Path to the image
img_path = './gen_pics/a_giant_robot_dragon.png'

def open_local_image(img_path):
    image = PIL.Image.open(img_path)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image

image = open_local_image(img_path)

prompt = "remove the background and put a moon in the sky"
result = pipe(prompt, image=image, num_inference_steps=200, image_guidance_scale=1).images
output_image = result[0]

# Save the output image
output_path = './gen_pics/output_image.png'
output_image.save(output_path)
