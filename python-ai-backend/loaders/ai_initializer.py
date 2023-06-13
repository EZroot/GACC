from diffusers import StableDiffusionPipeline, StableDiffusionControlNetInpaintPipeline, StableDiffusionControlNetPipeline, DPMSolverMultistepScheduler, ControlNetModel, DDIMScheduler
from diffusers.utils import load_image
import numpy as np
from utils.imageutils import save_file
import torch
from PIL import Image
from controlnet_aux import OpenposeDetector, LineartAnimeDetector


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

    print("Done...\n Scheduling: ", diffuser_model_id)

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    
    print("Done...\n CPU-Offload: ", use_cpu_offloading)
    if use_cpu_offloading:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to("cuda")
        pipe.enable_attention_slicing()
    return pipe

def initialize_diffusion_pipeline(diffuser_model_id, use_cpu_offloading=False):
    print("Loading Diffuser: ", diffuser_model_id)
    pipe = StableDiffusionPipeline.from_pretrained(diffuser_model_id, torch_dtype=torch.float16)

    print("Done...\n Scheduling...")

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    print("Done...\n CPU-Offload: ", use_cpu_offloading)
    if use_cpu_offloading:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to("cuda")
        pipe.enable_attention_slicing()
    return pipe

def generate_image_stablediffusion(pipe, prompt, negative_prompt, generator, height, width, num_inference_steps):
    image = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        generator=generator,
        height=height,
        width=width
    ).images[0]
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