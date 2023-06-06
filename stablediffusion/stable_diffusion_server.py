import os
import asyncio
from diffusers import StableDiffusionPipeline
import torch
from quart import Quart, jsonify, request
from PIL import Image
import json

semaphore = asyncio.Semaphore(1)  # Set the maximum number of concurrent requests to 1

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

# Load model ids from the JSON file
with open('model_ids.json', 'r') as file:
    model_ids = json.load(file)

# Prompt the user to choose a model id
print("Please choose a model id from the following options:")
for i, model_id in enumerate(model_ids, 1):
    print(f"{i}. {model_id}")

choice = int(input("Enter the number of your choice: "))
chosen_model_id = model_ids[choice - 1]  # -1 because list indices start at 0, but our printed options start at 1

print(f"You chose: {chosen_model_id}")


model_id = chosen_model_id
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe = pipe.to("cuda")

def dummy_checker(images, **kwargs):
    return images, False

pipe.safety_checker = dummy_checker

app = Quart(__name__)

async def generate_image_async(prompt, height, width, num_inference_steps, guidance_scale):
    result = await asyncio.to_thread(
        pipe,
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale
    )
    return result.images[0]

def load_config():
    script_path = os.path.abspath(__file__)
    script_directory = os.path.dirname(script_path)
    config_file = os.path.join(script_directory, 'config.json')

    with open(config_file) as f:
        config = json.load(f)
    return config

@app.route('/generateimg')
async def generate_image():
    async with semaphore:
        prompt = request.args.get('prompt', default='a photo of a banana on a pyramid', type=str)
        height = int(request.args.get('height', default=512, type=int))
        width = int(request.args.get('width', default=512, type=int))
        num_inference_steps = int(request.args.get('num_inference_steps', default=200, type=int))
        guidance_scale = float(request.args.get('guidance_scale', default=7.5, type=float))
        img_count = int(request.args.get('img_count', default=1, type=int))
        use_columns = bool(request.args.get('use_columns', default=True, type=bool))

        num_images = img_count
        results = []

        for i in range(num_images):
            image = await generate_image_async(prompt, height, width, num_inference_steps, guidance_scale)
            results.append(image)
            torch.cuda.empty_cache()  # Clear CUDA cache to release GPU memory

        num_columns = int(num_images ** 0.5)  # Calculate the number of columns for the grid
        num_rows = (num_images + num_columns - 1) // num_columns  # Calculate the number of rows for the grid

        grid = image_grid(results, rows=num_rows, cols=num_columns) if use_columns else image_grid(results, rows=num_images, cols=1)
        filename = f"./gen_pics/{prompt.replace(' ', '_').lower()}.png"
        os.makedirs('gen_pics', exist_ok=True)  # Create the directory if it doesn't exist
        await asyncio.to_thread(grid.save, filename)
        return jsonify({'image_path': os.path.abspath(filename)})


if __name__ == '__main__':
    config = load_config()
    debug = config.get('debug', False)
    port = config.get('port', 1337)
    app.run(debug=debug, port=port)
