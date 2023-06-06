# ultimate-ai-assistant
The ultimate AI assitant

Dont forget to create a config.json in the target/debug directory where the build is
```json
{
  "api_key": "token",
  "stable_diffusion_host_address": "localhost:6999",
  "python_code_create_iterations": 5,
  "python_use_conda_over_pip": true,
  "stable_diffusion_config": {
    "height": 512,
    "width": 712,
    "num_inference_steps": 800,
    "guidance_scale": 7.5,
    "img_count": 25,
    "use_columns": true
  }
}
```