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
    print("Done... \n Attempting to load lora weights [wowifierV3]")
    pipe.load_lora_weights("/mnt/c/Repos/ultimate-ai-assistant/python-ai-backend/models/loraweights/", weight_name="wowifierV3.safetensors")
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

def generate_image_stablediffusion(pipe, prompt, negative_prompt, generator, height, width, num_inference_steps,first_image_strength, resized_image_strength, first_image_noise,resized_image_noise):
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
    image_ref = resize(image,resized_image_noise)
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

def resize(image, noise):
    # Create a new image with the size 1024x1024
    #new_image = Image.new("RGB", (1344, 1344),(255,255,255))
    scaled_image = image.copy()
    scaled_image.resize((64,64))
    print("Extracting colors...")
    palette = extract_color_palette(scaled_image)
    print("Done...\n Generating random noise image (1344x1344)...")
    new_image = generate_random_noise_image((512,512), palette, 2) #scramble_colors_to_noise(image,1344,1344,noise) # add_noise_to_image_from_image(image, new_image,noise)
    new_image = new_image.resize((1344,1344))
    # Calculate the position to paste the image in the center
    position = ((1344 - image.width) // 2, (1344 - image.height) // 2)

    nw,nh = image.size
    print("Done...\n Scrambling the noise of the original a bit...")
    blurred_edge_image = scramble_colors_to_noise(image,nw,nh, noise)
    # Paste the image onto the new image
    new_image.save("before_new_image.png")
    new_image.paste(blurred_edge_image, position)
    new_image.save("after_new_image.png")
    # Return the resulting image
    return new_image

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

    # Perform k-means clustering to extract the dominant colors
    centroids, _ = kmeans(pixels, num_colors)

    # Create the color palette using the centroid values
    palette = (centroids * 255.0).astype(int).tolist()

    return palette



def generate_random_noise_image(size: tuple, palette: list, chunk_size: int) -> Image.Image:
    # Create a new image filled with random noise chunks using the color palette
    width, height = size
    num_colors = len(palette)

    # Calculate the number of chunks in each dimension
    num_chunks_x = width // chunk_size
    num_chunks_y = height // chunk_size

    # Generate random noise chunks
    random_chunks = np.random.choice(num_colors, size=(num_chunks_y, num_chunks_x))

    # Create an empty array for the final random noise image
    random_image_array = np.empty((height, width, 3), dtype=np.uint8)

    # Fill the random image array with random noise chunks
    for i in range(num_chunks_y):
        for j in range(num_chunks_x):
            chunk_color_index = random_chunks[i, j]
            chunk_color = palette[chunk_color_index]
            random_image_array[i * chunk_size: (i + 1) * chunk_size, j * chunk_size: (j + 1) * chunk_size] = chunk_color

    random_image = Image.fromarray(random_image_array, mode='RGB')

    return random_image