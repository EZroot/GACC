from diffusers import StableDiffusionPipeline,StableDiffusionImg2ImgPipeline, StableDiffusionLatentUpscalePipeline, StableDiffusionControlNetInpaintPipeline,KDPM2DiscreteScheduler, StableDiffusionControlNetPipeline, DPMSolverMultistepScheduler, ControlNetModel, DDIMScheduler
from diffusers.utils import load_image
from diffusers.models import AutoencoderKL
import numpy as np
from utils.imageutils import save_file
import torch
from PIL import Image,ImageFilter
from controlnet_aux import OpenposeDetector, LineartAnimeDetector
from scipy.ndimage import gaussian_filter
import random
from scipy.cluster.vq import kmeans
from joblib import Parallel, delayed
import os
from scipy.ndimage.filters import gaussian_filter
from enum import Enum

class CustomImageFilter(Enum):
    NO_FILTER = 0
    RANDOM_NOISE = 1
    GAUSSAIN_NOISE = 2
    RANDOM_NOISE_FROM_PALLETTE = 3

def initialize_controlnet_pipeline(controlnet_model_id, diffuser_model_id, useinpainting, use_cpu_offloading=False):

    print("Loading controlnet: ", controlnet_model_id)
    controlnet = ControlNetModel.from_pretrained(
        controlnet_model_id, torch_dtype=torch.float16
    )
    
    print("Done...\n Loading Diffuser: ", diffuser_model_id)
    print("Inpainting: ",useinpainting)
    if useinpainting:
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            diffuser_model_id, controlnet=controlnet, torch_dtype=torch.float16
        )
    else:
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            diffuser_model_id, controlnet=controlnet, torch_dtype=torch.float16
        )

    print("Done...\n Scheduling [DDIMScheduler]: ", diffuser_model_id)

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    
    print("Done...\n CPU-Offload: ", use_cpu_offloading)
    if use_cpu_offloading:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to("cuda")
        pipe.enable_attention_slicing()


    return pipe

def dummy_checker(images, **kwargs):
    return images, False

def initialize_diffusion_pipeline(diffuser_model_id, use_cpu_offloading=False):
    print("Loading Diffuser: ", diffuser_model_id)
    if "stable-diffusion-v1-5" in diffuser_model_id:
        pipe = StableDiffusionPipeline.from_pretrained(diffuser_model_id, torch_dtype=torch.float16, safety_checker=None)
    else:
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        pipe = StableDiffusionImg2ImgPipeline.from_ckpt(diffuser_model_id, vae=vae, torch_dtype=torch.float16)
        
        #controlnet = ControlNetModel.from_pretrained('lllyasviel/control_v11f1e_sd15_tile', torch_dtype=torch.float16)
        #pipe = StableDiffusionPipeline.from_ckpt(diffuser_model_id, torch_dtype=torch.float16, custom_pipeline="stable_diffusion_controlnet_img2img", controlnet=controlnet, safety_checker=None)

    print("Done...\n Disabling safety checker...")
    pipe.safety_checker = None
    print("Done...\n Scheduling...")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    #model_id = "stabilityai/sd-x2-latent-upscaler"

    #upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)

    #upscaler.to("cuda")

    
# #     pipeline.scheduler.compatibles
# # [
# #     diffusers.schedulers.scheduling_lms_discrete.LMSDiscreteScheduler,
# #     diffusers.schedulers.scheduling_unipc_multistep.UniPCMultistepScheduler,
# #     diffusers.schedulers.scheduling_k_dpm_2_discrete.KDPM2DiscreteScheduler,
# #     diffusers.schedulers.scheduling_deis_multistep.DEISMultistepScheduler,
# #     diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler,
# #     diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler,
# #     diffusers.schedulers.scheduling_ddpm.DDPMScheduler,
# #     diffusers.schedulers.scheduling_dpmsolver_singlestep.DPMSolverSinglestepScheduler,
# #     diffusers.schedulers.scheduling_k_dpm_2_ancestral_discrete.KDPM2AncestralDiscreteScheduler,
# #     diffusers.schedulers.scheduling_heun_discrete.HeunDiscreteScheduler,
# #     diffusers.schedulers.scheduling_pndm.PNDMScheduler,
# #     diffusers.schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteScheduler,
# #     diffusers.schedulers.scheduling_ddim.DDIMScheduler,
# # ]
    print("Done...\n CPU-Offload: ", use_cpu_offloading)
    if use_cpu_offloading:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to("cuda")
        pipe.enable_attention_slicing()

    print("Attempting to load lora weights [easynegative]")
    pipe.load_textual_inversion("/mnt/c/Repos/ultimate-ai-assistant/python-ai-backend/models/loraweights/", weight_name="easynegative.safetensors")
    #print("Done... \n Attempting to load lora weights [wowifierV3]")
    #pipe.load_lora_weights("/mnt/c/Repos/ultimate-ai-assistant/python-ai-backend/models/loraweights/", weight_name="wowifierV3.safetensors")
    print("Done... \n Attempting to load lora weights [more_details]")
    pipe.load_lora_weights("/mnt/c/Repos/ultimate-ai-assistant/python-ai-backend/models/loraweights/", weight_name="more_details.safetensors")
    # print("Done... \n Attempting to load lora weights [ElementalMagicAIv2-000008]")
    # pipe.load_lora_weights("/mnt/c/Repos/ultimate-ai-assistant/python-ai-backend/models/loraweights/", weight_name="ElementalMagicAIv2-000008.safetensors")
    # print("Done... \n Attempting to load lora weights [GamerFashion-rgb-V1]")
    # pipe.load_lora_weights("/mnt/c/Repos/ultimate-ai-assistant/python-ai-backend/models/loraweights/", weight_name="GamerFashion-rgb-V1.safetensors")
    # print("Done... \n Attempting to load lora weights [paseer]")
    # pipe.load_lora_weights("/mnt/c/Repos/ultimate-ai-assistant/python-ai-backend/models/loraweights/", weight_name="paseer.safetensors")
    #print("Done... \n Attempting to load lora weights [HarvestBeta]")
    #pipe.load_lora_weights("/mnt/c/Repos/ultimate-ai-assistant/python-ai-backend/models/loraweights/", weight_name="HarvestBeta.safetensors")
    # print("Done... \n Attempting to load lora weights [BioPunkAI]")
    # pipe.load_lora_weights("/mnt/c/Repos/ultimate-ai-assistant/python-ai-backend/models/loraweights/", weight_name="BioPunkAI.safetensors")
    # print("Done... \n Attempting to load lora weights [VampiricTech-20]")
    # pipe.load_lora_weights("/mnt/c/Repos/ultimate-ai-assistant/python-ai-backend/models/loraweights/", weight_name="VampiricTech-20.safetensors")
    print("Done... \n Attempting to load lora weights [negative_hand-neg]")
    pipe.load_textual_inversion("/mnt/c/Repos/ultimate-ai-assistant/python-ai-backend/models/pt/", weight_name="negative_hand-neg.pt")
    #print("Done... \n Attempting to load lora weights [ManyEyedHorrorAI-000011]")
    #pipe.load_lora_weights("/mnt/c/Repos/ultimate-ai-assistant/python-ai-backend/models/loraweights/", weight_name="ManyEyedHorrorAI-000011.safetensors")

    return pipe

def generate_image_stablediffusion(pipe, prompt, negative_prompt, generator, height,
                                width, num_inference_steps,first_image_strength, resized_image_strength, 
                                chunk_size,blur_radius,edge_radius, upscaled_size, first_image_noise, upscaled_image_filter_enum:CustomImageFilter, upscale_original:bool):
    image_ref = create_blank_image(width,height,(255,255,255))
    image_ref = add_noise_to_image(image_ref,first_image_noise)
    image = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        generator=generator,
        strength=first_image_strength,
        image=image_ref
    ).images[0]
    print("Resizing to 1024x1024")
    image_ref = resize(image,upscaled_size,upscaled_image_filter_enum, upscale_original, chunk_size,blur_radius,edge_radius)
    print("Clearing GPU cache...")
    torch.cuda.empty_cache()
    image = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        generator=generator,
        strength=resized_image_strength,
        image=image_ref
    ).images[0]
    # image = pipe(
    #     prompt,
    #     negative_prompt=negative_prompt,
    #     num_inference_steps=num_inference_steps,
    #     generator=generator,
    #     height=height,
    #     width=width
    # ).images[0]
    return image

def generate_image_controlnet_inpaint(pipe, generator, init_image, mask_image, control_image, prompt, negative_prompt, num_inference_steps):
    image = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        generator=generator,
        image=init_image,
        mask_image=mask_image,
        control_image=control_image,
    ).images[0]
    return image

def generate_image_controlnet_pic2pic(pipe, generator, pic2pic_image_filepath, prompt, negative_prompt, num_inference_steps):
    print(f"Attempting to load {pic2pic_image_filepath}")
    pic2pic_image = Image.open(pic2pic_image_filepath)
    print(f"Done...\n Processing image...")
    image = pipe(
        prompt,
        image=pic2pic_image,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        generator=generator,
    ).images[0]
    print(f"Done...")
    return image

def generate_image_controlnet_lineart(pipe, generator, lineart_image_filepath, prompt, negative_prompt, num_inference_steps):
    print(f"Attempting to load {lineart_image_filepath}")
    lineart_image = Image.open(lineart_image_filepath)
    print(f"Done...\n Loading processor: Anime Lineart")
    processor = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")
    print(f"Done...\n Processing image...")
    processed_image = processor(lineart_image)
    image = pipe(
        prompt,
        image=processed_image,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        generator=generator,
    ).images[0]
    print(f"Done...")
    return image

def generate_image_controlnet_open_pose(pipe, image_filepath, prompt, generator, negative_prompt, num_inference_steps):
    openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
    with Image.open(image_filepath).convert("RGB").resize((512, 768)) as image:
        open_pose_image = openpose(image, hand_and_face=True)
        save_file(open_pose_image, "open_pose_pose")
        image = openpose(image, hand_and_face=True)
        generated_image = pipe(
                prompt,
                negative_prompt=negative_prompt,
                image=image_filepath, 
                generator=generator, 
                num_inference_steps=num_inference_steps
                ).images[0]
        return generated_image

def create_blank_image(width, height, color=(0, 0, 0)):
    """
    Create a blank image with the given width, height, and RGBA color.
    """
    image = Image.new("RGB", (width, height), color)
    return image

def scramble_colors_to_noise(input_image_orig, new_width, new_height, noise_scale=25):
    # Load the input image
    input_image = input_image_orig.copy()
    input_image = input_image.resize((new_width, new_height))  # Resize input image

    # Create a new image with the desired dimensions
    noise_image = Image.new("RGB", (new_width, new_height))

    # Scramble the RGB values and generate random noise for each pixel
    for x in range(new_width):
        for y in range(new_height):
            r, g, b = input_image.getpixel((x, y))

            # Generate a random noise value for each color channel
            noise_r = random.randint(-noise_scale, noise_scale)
            noise_g = random.randint(-noise_scale, noise_scale)
            noise_b = random.randint(-noise_scale, noise_scale)

            # Calculate the new RGB values by scrambling the colors
            scrambled_r = (r + noise_r) % 256
            scrambled_g = (g + noise_g) % 256
            scrambled_b = (b + noise_b) % 256

            # Create a pixel with the scrambled RGB values
            noise_image.putpixel((x, y), (scrambled_r, scrambled_g, scrambled_b))
    noise_image.save("./scrambled_image.png")
    return noise_image

def add_noise_to_image_from_image(from_image, to_image, noise_scale=25):
    # Resize the "to" image to match the dimensions of the "from" image
    from_image_resized = from_image.resize(to_image.size)
    from_image_array = np.array(from_image_resized, dtype=np.float32)
    to_image_array = np.array(to_image, dtype=np.float32)
    # Step 2: Separate the images into color channels
    from_red_channel = from_image_array[:, :, 0]
    from_green_channel = from_image_array[:, :, 1]
    from_blue_channel = from_image_array[:, :, 2]
    to_red_channel = to_image_array[:, :, 0]
    to_green_channel = to_image_array[:, :, 1]
    to_blue_channel = to_image_array[:, :, 2]
    # Step 3: Generate separate noise for each color channel based on the "from" image
    red_noise = np.random.normal(loc=0, scale=noise_scale, size=from_red_channel.shape) * from_red_channel / 255.0
    green_noise = np.random.normal(loc=0, scale=noise_scale, size=from_green_channel.shape) * from_green_channel / 255.0
    blue_noise = np.random.normal(loc=0, scale=noise_scale, size=from_blue_channel.shape) * from_blue_channel / 255.0
    
    # Step 4: Add noise to each color channel of the "to" image
    noisy_to_red_channel = np.clip(to_red_channel + red_noise, 0, 255).astype(np.uint8)
    noisy_to_green_channel = np.clip(to_green_channel + green_noise, 0, 255).astype(np.uint8)
    noisy_to_blue_channel = np.clip(to_blue_channel + blue_noise, 0, 255).astype(np.uint8)
    
    # Step 5: Combine the noisy color channels of the "to" image
    noisy_to_image_array = np.stack((noisy_to_red_channel, noisy_to_green_channel, noisy_to_blue_channel), axis=2)
    
    # Step 6: Convert the array back to PIL image
    noisy_to_pil_image = Image.fromarray(noisy_to_image_array.astype(np.uint8))

    noisy_to_pil_image.save("./NoisyImage.png")
    return noisy_to_pil_image

def add_noise_to_image(image, noise_scale=25):
    # Step 1: Convert the image to a numerical format
    image_array = np.array(image, dtype=np.float32)

    # Step 2: Separate the image into color channels
    red_channel = image_array[:, :, 0]
    green_channel = image_array[:, :, 1]
    blue_channel = image_array[:, :, 2]

    # Step 3: Generate separate noise for each color channel
    red_noise = np.random.normal(loc=0, scale=noise_scale, size=red_channel.shape)
    green_noise = np.random.normal(loc=0, scale=noise_scale, size=green_channel.shape)
    blue_noise = np.random.normal(loc=0, scale=noise_scale, size=blue_channel.shape)

    # Step 4: Add noise to each color channel
    noisy_red_channel = np.clip(red_channel + red_noise, 0, 255).astype(np.uint8)
    noisy_green_channel = np.clip(green_channel + green_noise, 0, 255).astype(np.uint8)
    noisy_blue_channel = np.clip(blue_channel + blue_noise, 0, 255).astype(np.uint8)

    # Step 5: Combine the noisy color channels
    noisy_image_array = np.stack((noisy_red_channel, noisy_green_channel, noisy_blue_channel), axis=2)

    # Step 6: Convert the array back to PIL image
    noisy_pil_image = Image.fromarray(noisy_image_array)
    noisy_pil_image.save("./noise_added_to_blank.png")
    return noisy_pil_image

def resize(image,upscaled_size, upscaled_image_filter_enum:CustomImageFilter, upscale_original:bool, chunk_size=40, blur_radius=10, blur_radius_edge=6):
    # Create a new image with the size 1024x1024
    #new_image = Image.new("RGB", (1344, 1344),(255,255,255))
    scaled_image = image.copy()
    scaled_image.resize((64,64))
    if upscaled_image_filter_enum == CustomImageFilter.GAUSSAIN_NOISE:
        print("GAUSSAIN_NOISE FILTER ACTIVATED")
        print("Extracting colors...")
        palette = extract_color_palette(scaled_image)

        # Generate random noise image using the extracted palette
        size = image.size
        random_image = generate_random_noise_image(size, palette, chunk_size,blur_radius)

        # Display the random noise image
        random_image.save("randomized.png")
    elif upscaled_image_filter_enum == CustomImageFilter.RANDOM_NOISE:
        print("RANDOM_NOISE FILTER ACTIVATED")
        print("Extracting colors...")
        palette = extract_color_palette(scaled_image)

        # Generate random noise image using the extracted palette
        size = image.size
        random_image = add_noise_to_image(size, 500)

        # Display the random noise image
        random_image.save("randomized.png")
    elif upscaled_image_filter_enum == CustomImageFilter.NO_FILTER:
        print("NO FILTER ACTIVATED")
        # Generate random noise image using the extracted palette
        size = image.size
        width, height = size
        random_image = create_blank_image(width,height,(255,255,255))
        # Display the random noise image
        random_image.save("randomized.png")
    elif upscaled_image_filter_enum == CustomImageFilter.RANDOM_NOISE_FROM_PALLETTE:
        print("RANDOM_NOISE_FROM_PALLETTE FILTER ACTIVATED")
        print("Extracting colors...")
        palette = extract_color_palette(scaled_image)

        # Generate random noise image using the extracted palette
        size = image.size
        random_image = generate_random_noise_image(size, palette, 1,0)

        # Display the random noise image
        random_image.save("randomized.png")

    if upscale_original:
        # Upscale and paste the original image
        upscaled_and_pasted_image = upscale_and_paste(random_image, image, blur_radius_edge, upscaled_size, upscaled_size)
    else:
        upscaled_and_pasted_image = upscale_and_paste(random_image, image, blur_radius_edge, upscaled_size, image.size)

    # Save the final image
    upscaled_and_pasted_image.save("final_image.png")

    # Return the resulting image
    return upscaled_and_pasted_image

def add_edge_noise(image, blur_radius, noise_scale):
    # Apply a Gaussian blur to the image
    blurred_image = image.filter(ImageFilter.GaussianBlur(blur_radius))
    
    # Convert the blurred image to a numpy array
    blurred_array = np.array(blurred_image)
    
    # Generate random noise with the same shape as the blurred image
    noise = np.random.normal(loc=0, scale=noise_scale, size=blurred_array.shape)
    
    # Add the noise to the blurred image array
    noisy_blurred_array = np.clip(blurred_array + noise, 0, 255).astype(np.uint8)
    
    # Convert the noisy blurred array back to an image
    noisy_blurred_image = Image.fromarray(noisy_blurred_array)
    
    # Composite the noisy blurred image with the original image
    composited_image = Image.blend(image, noisy_blurred_image, 0.5)
    
    return composited_image

def extract_color_palette(image: Image.Image, num_colors: int = 256) -> list:
    # Convert the image to an array of RGB values
    image_array = np.array(image)

    # Normalize RGB values
    pixels = image_array.reshape(-1, 3) / 255.0

    def compute_kmeans(pixels, num_colors):
        # Perform k-means clustering to extract the dominant colors
        centroids, _ = kmeans(pixels, num_colors)
        return (centroids * 255.0).astype(int).tolist()

    # Split the pixels into chunks for parallel processing
    num_pixels = pixels.shape[0]
    chunk_size = num_pixels // num_colors
    pixel_chunks = [pixels[i:i+chunk_size] for i in range(0, num_pixels, chunk_size)]

    # Determine the number of threads based on the image size
    num_threads = os.cpu_count()
    if num_threads is None or num_threads < 1:
        num_threads = 1

    # Run k-means clustering in parallel
    results = Parallel(n_jobs=num_threads)(delayed(compute_kmeans)(chunk, num_colors) for chunk in pixel_chunks)

    # Concatenate the results from all chunks
    palette = np.concatenate(results).tolist()

    return palette


def generate_random_noise_image(size: tuple, palette: list, chunk_size: int, blur_radius: float) -> Image.Image:
    # Create a new image filled with random noise chunks using the color palette
    width, height = size
    num_colors = len(palette)

    # Calculate the number of chunks in each dimension
    num_chunks_x = (width + chunk_size - 1) // chunk_size  # Round up the division
    num_chunks_y = (height + chunk_size - 1) // chunk_size  # Round up the division

    # Generate random noise chunks with a Gaussian distribution
    mean = num_colors // 2  # Mean value for Gaussian distribution
    std = num_colors // 4  # Standard deviation for Gaussian distribution
    random_chunks = np.random.normal(mean, std, size=(num_chunks_y, num_chunks_x)).astype(int)

    # Create an empty array for the final random noise image
    random_image_array = np.empty((height, width, 3), dtype=np.uint8)

    # Fill the random image array with random noise chunks
    for i in range(num_chunks_y):
        for j in range(num_chunks_x):
            chunk_color_index = random_chunks[i, j] % num_colors
            chunk_color = palette[chunk_color_index]
            x_start = j * chunk_size
            y_start = i * chunk_size
            x_end = min(x_start + chunk_size, width)
            y_end = min(y_start + chunk_size, height)
            random_image_array[y_start:y_end, x_start:x_end] = chunk_color

    # Create a copy of the random image array for blurring
    blurred_image_array = np.copy(random_image_array)

    # Apply Gaussian blur to each color channel separately
    for c in range(3):
        blurred_image_array[:, :, c] = gaussian_filter(blurred_image_array[:, :, c], sigma=blur_radius)

    blurred_image = Image.fromarray(blurred_image_array, mode='RGB')

    return blurred_image

def upscale_and_paste(image: Image.Image, original_image: Image.Image, blur_radius: float, upscaled_size, original_upscaled_size) -> Image.Image:
    # Upscale the image to twice its size
    original_image= original_image.resize(original_upscaled_size, Image.BICUBIC)
    new_width, new_height = upscaled_size
    upscaled_image = image.resize((new_width, new_height), Image.BICUBIC)

    # Calculate the coordinates to paste the original image in the center
    paste_x = (upscaled_image.width - original_image.width) // 2
    paste_y = (upscaled_image.height - original_image.height) // 2

    # Create a blurred version of the original image
    #blurred_image = original_image.filter(ImageFilter.GaussianBlur(blur_radius))

    # Create edge masks for the top, bottom, left, and right edges
    edge_mask = Image.new("L", original_image.size, 0)
    edge_width = int(original_image.width * 0.1)  # Adjust the width ratio as desired
    edge_height = int(original_image.height * 0.1)  # Adjust the height ratio as desired
    edge_mask.paste(255, (0, 0, original_image.width, edge_height))  # Top edge
    edge_mask.paste(255, (0, original_image.height - edge_height, original_image.width, original_image.height))  # Bottom edge
    edge_mask.paste(255, (0, 0, edge_width, original_image.height))  # Left edge
    edge_mask.paste(255, (original_image.width - edge_width, 0, original_image.width, original_image.height))  # Right edge

    # Apply Gaussian blur to the edge masks
    blurred_edge_mask = edge_mask.filter(ImageFilter.GaussianBlur(blur_radius))

    # Invert the blurred edge masks
    inverted_blurred_edge_mask = Image.eval(blurred_edge_mask, lambda x: 255 - x)

    # Create a new image to hold the final result
    final_image = Image.new("RGB", upscaled_image.size)

    # Paste the upscaled image onto the final image
    final_image.paste(upscaled_image, (0, 0))

    # Paste the blurred image in the center
    final_image.paste(original_image, (paste_x, paste_y), mask=inverted_blurred_edge_mask)

    return final_image

