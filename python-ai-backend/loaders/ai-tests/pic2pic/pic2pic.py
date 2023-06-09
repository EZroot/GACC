import PIL
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float32, safety_checker=None)
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# Path to the image
img_path = './gen_pics/lion.png'

def open_local_image(img_path):
    image = PIL.Image.open(img_path)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image

image = open_local_image(img_path)

prompt = "change the hand to a paw"
result = pipe(prompt, image=image, num_inference_steps=50, image_guidance_scale=0).images
output_image = result[0]

# Save the output image
output_path = './gen_pics/lion_fixed.png'
output_image.save(output_path)
