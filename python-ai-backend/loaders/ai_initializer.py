from diffusers import StableDiffusionPipeline,StableDiffusionImg2ImgPipeline, StableDiffusionLatentUpscalePipeline, StableDiffusionControlNetInpaintPipeline,KDPM2DiscreteScheduler, StableDiffusionControlNetPipeline, DPMSolverMultistepScheduler, ControlNetModel, DDIMScheduler
from diffusers.utils import load_image
from diffusers.models import AutoencoderKL
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

    model_id = "stabilityai/sd-x2-latent-upscaler"

    upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)

    upscaler.to("cuda")

    
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

    return upscaler,pipe

def generate_image_stablediffusion(upscaler, pipe, prompt, negative_prompt, generator, height, width, num_inference_steps):
    low_res_latents = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        generator=generator,
        height=height,
        width=width,
        output_type="latent"
    ).images[0]
    torch.cuda.empty_cache()
    image = upscaler(
    prompt=prompt,
    image=low_res_latents,
    num_inference_steps=50,
    guidance_scale=0,
    generator=generator,
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