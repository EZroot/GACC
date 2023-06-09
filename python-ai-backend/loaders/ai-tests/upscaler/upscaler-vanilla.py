from diffusers import StableDiffusionLatentUpscalePipeline, StableDiffusionPipeline
import torch
from PIL import Image

upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained("stabilityai/sd-x2-latent-upscaler", torch_dtype=torch.float16)
upscaler.to("cuda")
upscaler.enable_attention_slicing() 

prompt = "high resolution, ultra realistic, focused"
generator = torch.manual_seed(33)

# we stay in latent space! Let's make sure that Stable Diffusion returns the image
# in latent space
image_path = "./gen_pics/a_giant_robot_dragon.png"
img = Image.open(image_path).convert("RGB")

upscaled_image = upscaler(
    prompt=prompt,
    image=img,
    num_inference_steps=200,
    guidance_scale=0,
    generator=generator,
).images[0]

# Let's save the upscaled image under "upscaled_astronaut.png"
upscaled_image.save("./gen_pics/upscaled.png")

