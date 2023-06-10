from loaders.ai_generator import request_generate_image_openpose, request_generate_image_stablediffusion
from loaders.ai_model_selection import get_ai_model
from loaders.ai_initializer import generate_image_controlnet_inpaint, generate_image_controlnet_open_pose, generate_image_stablediffusion, initialize_controlnet_pipeline, initialize_diffusion_pipeline
from quart import Quart, jsonify, request

app = Quart(__name__)

ai_model = get_ai_model()

if ai_model.controlnet_model_id != None:
    pipe = initialize_controlnet_pipeline(ai_model.controlnet_model_id, ai_model.diffusion_model_id, ai_model.useinpainting)
    use_control_net = True
else:
    pipe = initialize_diffusion_pipeline(ai_model.diffusion_model_id)
    use_control_net = False

@app.route(ai_model.route)
async def generate_image_endpoint():
    args = request.args
    if use_control_net:
        response = request_generate_image_openpose(pipe, args)
    else:
        response = request_generate_image_stablediffusion(pipe, args)
    return response

if __name__ == '__main__':
    ai_model.display_models()
    print("Using Controlnet: ",use_control_net)
    app.run(debug=False, port=6969)