import json
from loaders.data.ai_model import AIModel

import json

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

def _prompt_and_select_model(models):
    config = load_config('ai_config.json')
    
    controlnet_models = config['AI']['controlnet']['models']
    diffusion_models = config['AI']['diffusion']['models']

    # Display the available models
    print("Control-Net Models:")
    for i, model in enumerate(controlnet_models, start=1):
        print(f"{i}. {model['model']}")

    print("\nDiffusion Models:")
    for i, model in enumerate(diffusion_models, start=1):
        print(f"{i}. {model}")

    choice = input("Enter the number of the model you want to pick: ")

    # Validate the model choice
    if not choice.isdigit() or int(choice) < 1 or int(choice) > len(models):
        print("Invalid model choice!")
        exit()

    selected_model = models[int(choice) - 1]
    return selected_model

def get_ai_model():
    config = load_config('ai_config.json')
    controlnet_models = config['AI']['controlnet']['models']
    diffusion_models = config['AI']['diffusion']['models']

    model_choice = input("Enter 'C' for Control-Net or 'D' for Stable Diffusion: ")

    if model_choice.upper() == 'C':
        print("\nControl-Net Models:")
        for i, model in enumerate(controlnet_models, start=1):
            print(f"{i}. {model['model']}")

        controlnet_choice = input("Enter the number of the Control-Net model you want to pick: ")

        if not controlnet_choice.isdigit() or int(controlnet_choice) < 1 or int(controlnet_choice) > len(controlnet_models):
            print("Invalid Control-Net model choice!")
            exit()

        selected_controlnet_model = controlnet_models[int(controlnet_choice) - 1]['model']
        use_inpainting = controlnet_models[int(controlnet_choice) - 1]['useInpainting']
        route = controlnet_models[int(controlnet_choice) - 1]['route']
        selected_diffusion_model = _prompt_and_select_model(diffusion_models)

    elif model_choice.upper() == 'D':
        selected_controlnet_model = None
        selected_diffusion_model = _prompt_and_select_model(diffusion_models)
        use_inpainting = False
        route = "/stablediffusion"
    else:
        print("Invalid model choice!")
        exit()

    return AIModel(selected_controlnet_model, selected_diffusion_model, use_inpainting, route)
