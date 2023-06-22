import os
import torch
from loaders.ai_initializer import generate_image_controlnet_lineart, generate_image_controlnet_open_pose, generate_image_controlnet_pic2pic, generate_image_stablediffusion
from utils.imageutils import save_file
from utils.imageutils import image_grid
from quart import Quart, jsonify, request


def request_generate_image_stablediffusion(pipe, request_args):
    prompt = request_args.get(
        'prompt', default='a photo of a banana on a pyramid', type=str)
    height = int(request_args.get('height', default=768, type=int))
    width = int(request_args.get('width', default=768, type=int))
    num_inference_steps = int(request_args.get(
        'num_inference_steps', default=50, type=int))
    img_count = int(request_args.get('img_count', default=1, type=int))
    use_columns = bool(request_args.get(
        'use_columns', default=True, type=bool))
    negative_prompt = request_args.get(
        'negative_prompt', default='deformed iris, deformed pupils, mutated hands and fingers:1.4), (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, duplicate, amputation, easynegative, negative_hand', type=str)

# first_image_strength, resized_image_strength, first_image_noise,resized_image_noise
    first_image_strength = request_args.get(
        'first_image_strength', default=1.0, type=float)
    resized_image_strength = request_args.get(
        'resized_image_strength', default=.8, type=float)
    chunk_size = request_args.get(
        'chunk_size', default=50, type=int)
    blur_radius = request_args.get(
        'blur_radius', default=10, type=int)
    edge_radius = request_args.get(
        'edge_radius', default=6, type=int)
    
    upscaled_size_width = request_args.get(
        'upscaled_size_width', default=512, type=int)
    upscaled_size_height = request_args.get(
        'upscaled_size_height', default=512, type=int)
# deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation, easynegative, bad-hands-5
    # Logging the received values for debugging
    print("Received values:")
    print("prompt:", prompt)
    print("num_inference_steps:", num_inference_steps)
    print("img_count:", img_count)
    print("use_columns:", use_columns)

    print("first_image_strength:", first_image_strength)
    print("resized_image_strength:", resized_image_strength)
    print("chunk_size:", chunk_size)
    print("blur_radius:", blur_radius)
    print("edge_radius:", edge_radius)

    num_images = img_count
    results = []
    for i in range(num_images):
        generator = torch.cuda.seed()
        image = generate_image_stablediffusion(pipe, prompt, negative_prompt, generator, height, width, num_inference_steps,
                                               first_image_strength, resized_image_strength, chunk_size, blur_radius, edge_radius, (upscaled_size_width,upscaled_size_height))
        results.append(image)
        torch.cuda.empty_cache()  # Clear CUDA cache to release GPU memory

    if num_images == 2:
        num_columns = 2
        num_rows = 1
    else:
        # Calculate the number of columns for the grid
        num_columns = int(num_images ** 0.5) if use_columns else img_count
        # Calculate the number of rows for the grid
        num_rows = (num_images + num_columns -
                    1) // num_columns if use_columns else 1

    grid = image_grid(results, rows=num_rows, cols=num_columns)
    filename = save_file(grid, prompt)
    return jsonify({'image_path': os.path.abspath(filename)})


def request_generate_image_pic2pic(pipe, request_args):
    prompt = request_args.get(
        'prompt', default='a photo of a banana on a pyramid', type=str)
    negative_prompt = request_args.get(
        'negative_prompt', default='cropped, lowres, poorly drawn face, out of frame, poorly drawn hands, blurry, bad art, blurred, text, watermark, disfigured, deformed, closed eyes', type=str)
    filepath_pose = request_args.get('filepath', default='derp.png', type=str)
    num_inference_steps = int(request_args.get(
        'num_inference_steps', default=50, type=int))
    img_count = int(request_args.get('img_count', default=1, type=int))
    use_columns = bool(request_args.get(
        'use_columns', default=True, type=bool))

    num_images = img_count
    results = []
    for i in range(num_images):
        generator = torch.cuda.seed()
        image = generate_image_controlnet_pic2pic(
            pipe, generator, filepath_pose, prompt, negative_prompt, num_inference_steps)
        results.append(image)
        torch.cuda.empty_cache()  # Clear CUDA cache to release GPU memory

    if num_images == 2:
        num_columns = 2
        num_rows = 1
    else:
        # Calculate the number of columns for the grid
        num_columns = int(num_images ** 0.5) if use_columns else img_count
        # Calculate the number of rows for the grid
        num_rows = (num_images + num_columns -
                    1) // num_columns if use_columns else 1

    grid = image_grid(results, rows=num_rows, cols=num_columns)
    filename = save_file(grid, prompt)
    return jsonify({'image_path': os.path.abspath(filename)})


def request_generate_image_lineart(pipe, request_args):
    prompt = request_args.get(
        'prompt', default='a photo of a banana on a pyramid', type=str)
    negative_prompt = request_args.get(
        'negative_prompt', default='cropped, lowres, poorly drawn face, out of frame, poorly drawn hands, blurry, bad art, blurred, text, watermark, disfigured, deformed, closed eyes', type=str)
    filepath_pose = request_args.get('filepath', default='derp.png', type=str)
    num_inference_steps = int(request_args.get(
        'num_inference_steps', default=50, type=int))
    img_count = int(request_args.get('img_count', default=1, type=int))
    use_columns = bool(request_args.get(
        'use_columns', default=True, type=bool))

    num_images = img_count
    results = []
    for i in range(num_images):
        generator = torch.cuda.seed()
        image = generate_image_controlnet_lineart(
            pipe, generator, filepath_pose, prompt, negative_prompt, num_inference_steps)
        results.append(image)
        torch.cuda.empty_cache()  # Clear CUDA cache to release GPU memory

    if num_images == 2:
        num_columns = 2
        num_rows = 1
    else:
        # Calculate the number of columns for the grid
        num_columns = int(num_images ** 0.5) if use_columns else img_count
        # Calculate the number of rows for the grid
        num_rows = (num_images + num_columns -
                    1) // num_columns if use_columns else 1

    grid = image_grid(results, rows=num_rows, cols=num_columns)
    filename = save_file(grid, prompt)
    return jsonify({'image_path': os.path.abspath(filename)})


def request_generate_image_openpose(pipe, request_args):
    prompt = request_args.get(
        'prompt', default='a photo of a banana on a pyramid', type=str)
    negative_prompt = request_args.get(
        'negative_prompt', default='cropped, lowres, poorly drawn face, out of frame, poorly drawn hands, blurry, bad art, blurred, text, watermark, disfigured, deformed, closed eyes', type=str)
    filepath_pose = request_args.get('filepath', default='derp.png', type=str)
    num_inference_steps = int(request_args.get(
        'num_inference_steps', default=50, type=int))
    img_count = int(request_args.get('img_count', default=1, type=int))
    use_columns = bool(request_args.get(
        'use_columns', default=True, type=bool))

    num_images = img_count
    results = []
    for i in range(num_images):
        generator = torch.cuda.seed()
        image = generate_image_controlnet_open_pose(
            pipe, filepath_pose, prompt, generator, negative_prompt, num_inference_steps)
        results.append(image)
        torch.cuda.empty_cache()  # Clear CUDA cache to release GPU memory

    if num_images == 2:
        num_columns = 2
        num_rows = 1
    else:
        # Calculate the number of columns for the grid
        num_columns = int(num_images ** 0.5) if use_columns else img_count
        # Calculate the number of rows for the grid
        num_rows = (num_images + num_columns -
                    1) // num_columns if use_columns else 1

    grid = image_grid(results, rows=num_rows, cols=num_columns)
    filename = save_file(grid, prompt)
    return jsonify({'image_path': os.path.abspath(filename)})
