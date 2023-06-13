import os
import torch
from loaders.ai_initializer import generate_image_controlnet_lineart, generate_image_controlnet_open_pose, generate_image_stablediffusion
from utils.imageutils import save_file
from utils.imageutils import image_grid
from quart import Quart, jsonify, request

def request_generate_image_stablediffusion(pipe, request_args):
    prompt = request_args.get('prompt', default='a photo of a banana on a pyramid', type=str)
    height = int(request_args.get('height', default=768, type=int))
    width = int(request_args.get('width', default=768, type=int))
    num_inference_steps = int(request_args.get('num_inference_steps', default=50, type=int))
    img_count = int(request_args.get('img_count', default=1, type=int))
    use_columns = bool(request_args.get('use_columns', default=True, type=bool))
    negative_prompt = request_args.get('negative_prompt', default='cropped, lowres, poorly drawn face, out of frame, poorly drawn hands, blurry, bad art, blurred, text, watermark, disfigured, deformed, closed eyes', type=str)

       # Logging the received values for debugging
    print("Received values:")
    print("prompt:", prompt)
    print("num_inference_steps:", num_inference_steps)
    print("img_count:", img_count)
    print("use_columns:", use_columns)
    
    num_images = img_count
    results = []
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

def request_generate_image_openpose(pipe, request_args):
    prompt = request_args.get('prompt', default='a photo of a banana on a pyramid', type=str)
    negative_prompt = request_args.get('negative_prompt', default='cropped, lowres, poorly drawn face, out of frame, poorly drawn hands, blurry, bad art, blurred, text, watermark, disfigured, deformed, closed eyes', type=str)
    filepath_pose = request_args.get('filepath', default='derp.png', type=str)
    num_inference_steps = int(request_args.get('num_inference_steps', default=50, type=int))
    img_count = int(request_args.get('img_count', default=1, type=int))
    use_columns = bool(request_args.get('use_columns', default=True, type=bool))

    num_images = img_count
    results = []
    for i in range(num_images):
        generator = torch.cuda.seed()
        image = generate_image_controlnet_open_pose(pipe, filepath_pose, prompt, generator, negative_prompt, num_inference_steps)
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

def request_generate_image_lineart(pipe, request_args):
    prompt = request_args.get('prompt', default='a photo of a banana on a pyramid', type=str)
    negative_prompt = request_args.get('negative_prompt', default='cropped, lowres, poorly drawn face, out of frame, poorly drawn hands, blurry, bad art, blurred, text, watermark, disfigured, deformed, closed eyes', type=str)
    filepath_pose = request_args.get('filepath', default='derp.png', type=str)
    num_inference_steps = int(request_args.get('num_inference_steps', default=50, type=int))
    img_count = int(request_args.get('img_count', default=1, type=int))
    use_columns = bool(request_args.get('use_columns', default=True, type=bool))

    num_images = img_count
    results = []
    for i in range(num_images):
        generator = torch.cuda.seed()
        image = generate_image_controlnet_lineart(pipe, generator, filepath_pose, prompt, negative_prompt, num_inference_steps)
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

def request_generate_image_openpose(pipe, request_args):
    prompt = request_args.get('prompt', default='a photo of a banana on a pyramid', type=str)
    negative_prompt = request_args.get('negative_prompt', default='cropped, lowres, poorly drawn face, out of frame, poorly drawn hands, blurry, bad art, blurred, text, watermark, disfigured, deformed, closed eyes', type=str)
    filepath_pose = request_args.get('filepath', default='derp.png', type=str)
    num_inference_steps = int(request_args.get('num_inference_steps', default=50, type=int))
    img_count = int(request_args.get('img_count', default=1, type=int))
    use_columns = bool(request_args.get('use_columns', default=True, type=bool))

    num_images = img_count
    results = []
    for i in range(num_images):
        generator = torch.cuda.seed()
        image = generate_image_controlnet_open_pose(pipe, filepath_pose, prompt, generator, negative_prompt, num_inference_steps)
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