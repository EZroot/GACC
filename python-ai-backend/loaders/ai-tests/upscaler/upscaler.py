from PIL import Image
import torch
import numpy as np
from diffusers import StableDiffusionUpscalePipeline

# load model and scheduler
model_id = "stabilityai/stable-diffusion-x4-upscaler"
pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float32)
torch.backends.cuda.max_split_size_mb = 256
pipeline = pipeline.to("cuda")
torch.cuda.empty_cache()

# Load the image from local file
image_path = "./gen_pics/a_giant_robot_dragon.png"
img = Image.open(image_path).convert("RGB")

# Define the number of chunks in each dimension
num_chunks_width = 4
num_chunks_height = 4

# Divide the image into chunks
width, height = img.size
chunk_width, chunk_height = width // num_chunks_width, height // num_chunks_height
chunks = []
indices = []
for i in range(0, width, chunk_width):
    for j in range(0, height, chunk_height):
        chunk = img.crop((i, j, i + chunk_width, j + chunk_height))
        chunks.append(chunk)
        indices.append((j // chunk_height, i // chunk_width))

upscaled_chunks = []
for low_res_img in chunks:
    # Downscale and upscale each chunk
    low_res_img = low_res_img.resize((128, 128))
    prompt = "high detailed"
    upscaled_chunk = pipeline(prompt=prompt, image=low_res_img).images[0]
    torch.cuda.empty_cache()
    upscaled_chunks.append(upscaled_chunk)

# Stitch the chunks back together
final_img = Image.new('RGB', (width*num_chunks_width, height*num_chunks_height))  # Considering that your pipeline quadruples the resolution
for idx, img in zip(indices, upscaled_chunks):
    final_img.paste(img, (idx[1]*width, idx[0]*height))

final_img.save("gen_pics/upscaled_image.png")
