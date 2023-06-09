from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from controlnet_aux import OpenposeDetector, PidiNetDetector, HEDdetector,LineartAnimeDetector
from diffusers.utils import load_image
from quart import Quart, request
import asyncio
import urllib.parse

app = Quart(__name__)

checkpoint = "lllyasviel/control_v11p_sd15s2_lineart_anime"

processor = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")

controlnet = ControlNetModel.from_pretrained(
    checkpoint, torch_dtype=torch.float16
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V1.4", controlnet=controlnet, torch_dtype=torch.float16
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")
pipe.enable_attention_slicing()

async def generate_image(prompt):
    try:
        with Image.open('./snapshots/image.png').convert("RGB").resize((512, 768)) as image:
            image = processor(image)
            image.save("./gen_pics/lineart_anime_control.png")
            torch.cuda.empty_cache()  # Clear CUDA cache to release GPU memory
            #generator = torch.Generator('cuda').manual_seed(1337)
            generated_image = pipe(prompt, image=image, negative_prompt='cropped, nudity, lowres, out of frame, too many fingers, blurry, bad art, blurred, text, watermark, disfigured, deformed').images[0]
            torch.cuda.empty_cache()  # Clear CUDA cache to release GPU memory
            generated_image.save(f'./gen_pics/newpic_lineart_anime_nicks.png')
            print('Image generated and saved')
            return generated_image
    except FileNotFoundError as e:
        print(f'Error: File not found: {str(e)}')
    except Exception as e:
        print(f'Error generating image: {str(e)}')

@app.route('/')
async def generate_and_save_image():
    generated_image = await generate_image("an beautiful girl, side view, smoke background,  vivid, high detail, studio gibli style")
    if generated_image is not None:
        return 'Image generated and saved'
    else:
        return 'Error generating image'

if __name__ == '__main__':
    app.run()
