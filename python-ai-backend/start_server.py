import os
from utils.imageutils import image_grid, save_file
import torch
from loaders.ai_model_selection import get_ai_model
from loaders.ai_initializer import generate_image_controlnet_inpaint, generate_image_controlnet_open_pose, generate_image_stablediffusion, initialize_controlnet_pipeline, initialize_diffusion_pipeline
from quart import Quart, jsonify, request
from PIL import Image

app = Quart(__name__)

ai_model = get_ai_model()
ai_model.display_models()

if ai_model.controlnet_model_id != None:
    pipe = initialize_controlnet_pipeline(ai_model.controlnet_model_id, ai_model.diffusion_model_id, ai_model.useinpainting)
else:
    pipe = initialize_diffusion_pipeline(ai_model.diffusion_model_id)

@app.route('/stablediffusion')
def generate_image_stablediffusion():
    prompt = request.args.get('prompt', default='a photo of a banana on a pyramid', type=str)
    height = int(request.args.get('height', default=512, type=int))
    width = int(request.args.get('width', default=512, type=int))
    num_inference_steps = int(request.args.get('num_inference_steps', default=200, type=int))
    guidance_scale = float(request.args.get('guidance_scale', default=7.5, type=float))
    img_count = int(request.args.get('img_count', default=1, type=int))
    use_columns = bool(request.args.get('use_columns', default=True, type=bool))

    num_images = img_count
    results = []
    negative_prompt = "cropped, lowres, poorly drawn face, out of frame, poorly drawn hands, blurry, bad art, blurred, text, watermark, disfigured, deformed, closed eyes"
    for i in range(num_images):
        generator = torch.cuda.seed()
        image = generate_image_stablediffusion(pipe, prompt, negative_prompt, generator, height, width, num_inference_steps)
        results.append(image)
        torch.cuda.empty_cache()  # Clear CUDA cache to release GPU memory

    if num_images == 2:
        num_columns = 2
        num_rows = 1
    else:
        num_columns = int(num_images ** 0.5) if use_columns else img_count  # Calculate the number of columns for the grid
        num_rows = (num_images + num_columns - 1) // num_columns if use_columns else 1  # Calculate the number of rows for the grid

    grid = image_grid(results, rows=num_rows, cols=num_columns)
    filename = save_file(grid, prompt)
    return jsonify({'image_path': os.path.abspath(filename)})

if __name__ == '__main__':
    app.run(debug=False, port=6969)