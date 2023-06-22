from typing import Callable, List, Optional, Union
from PIL import Image
import PIL
import numpy as np
import torch

from diffusers import StableDiffusionImg2ImgPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import preprocess

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    revision="fp16",
    torch_dtype=torch.float16,
)

pipe.set_use_memory_efficient_attention_xformers(True)
pipe.to("cuda")
# %%
# load the image, extract the mask
rgba = Image.open('tes_alphaedt.png')
#rgba = img.convert('RGBA')
mask_full = np.array(rgba)[:, :, 3] == 0
rgb = rgba.convert('RGB')
# %%

# resize/convert the mask to the right size
# for 512x512, the mask should be 1x4x64x64
hw = np.array(mask_full.shape)
h, w = (hw - hw % 32) // 8
mask_image = Image.fromarray(mask_full).resize((w, h), Image.NEAREST)
mask = (np.array(mask_image) == 0)[None, None]
mask = np.concatenate([mask]*4, axis=1)
mask = torch.from_numpy(mask).to('cuda')
mask.shape

# %%


@torch.no_grad()
def outpaint(
    self: StableDiffusionImg2ImgPipeline,
    prompt: Union[str, List[str]] = None,
    image: Union[torch.FloatTensor, PIL.Image.Image] = None,
    strength: float = 0.8,
    num_inference_steps: Optional[int] = 50,
    guidance_scale: Optional[float] = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    eta: Optional[float] = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    callback_steps: Optional[int] = 1,
    **kwargs,
):
    r"""
    copy of the original img2img pipeline's __call__()
    https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_img2img.py
    Changes are marked with <EDIT> and </EDIT>
    """
    # message = "Please use `image` instead of `init_image`."
    # init_image = deprecate("init_image", "0.14.0", message, take_from=kwargs)
    # image = init_image or image

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(prompt, strength, callback_steps,
                      negative_prompt, prompt_embeds, negative_prompt_embeds)

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]
    device = self._execution_device
    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0

    # 3. Encode input prompt
    prompt_embeds = self._encode_prompt(
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
    )

    # 4. Preprocess image
    image = preprocess(image)

    # 5. set timesteps
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps, num_inference_steps = self.get_timesteps(
        num_inference_steps, strength, device)
    latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

    # 6. Prepare latent variables
    latents = self.prepare_latents(
        image, latent_timestep, batch_size, num_images_per_prompt, prompt_embeds.dtype, device, generator
    )

    # <EDIT>
    # store the encoded version of the original image to overwrite
    # what the UNET generates "underneath" our image on each step
    encoded_original = (self.vae.config.scaling_factor *
                  self.vae.encode(
                      image.to(latents.device, latents.dtype)
                  ).latent_dist.mean)
    # </EDIT>

    # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    # 8. Denoising loop
    num_warmup_steps = len(timesteps) - \
        num_inference_steps * self.scheduler.order
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat(
                [latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t,
                                   encoder_hidden_states=prompt_embeds).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * \
                    (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs).prev_sample

            # <EDIT> paste unmasked regions from the original image
            noise = torch.randn(
                encoded_original.shape, generator=generator, device=device)
            noised_encoded_original = self.scheduler.add_noise(
                encoded_original, noise, t).to(noise_pred.device, noise_pred.dtype)
            latents[mask] = noised_encoded_original[mask]
            # </EDIT>

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

    # 9. Post-processing
    image = self.decode_latents(latents)

    # 10. Run safety checker
    image, has_nsfw_concept = self.run_safety_checker(
        image, device, prompt_embeds.dtype)

    # 11. Convert to PIL
    if output_type == "pil":
        image = self.numpy_to_pil(image)

    if not return_dict:
        return (image, has_nsfw_concept)

    return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)


# %%
image = outpaint(
    pipe,
    image=rgb,
    prompt="forest in the style of Tim Hildebrandt",
    strength=0.9,
    num_inference_steps=50,
    guidance_scale=7.5,
).images[0]

image.save("newimg.png")
