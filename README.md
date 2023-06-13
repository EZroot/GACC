# ultimate-ai-assistant
The ultimate AI assitant

[Rust App]
Pico Picasso - Image editor + AI Gen
* Draw and generate
* Regenerate images
* Edit images and regenerate regions
![Pico Picasso](https://cdn.discordapp.com/attachments/1051535103223930984/1118021969200296016/image.png)
![Pico Picasso](https://cdn.discordapp.com/attachments/1112415182975676497/1118063562926071808/image.png)

[Copilot]
Copilot - Code Generation AI
* automatically generates syntax passing files
* usage  ```./copilot -pycode generate a file to do something```

requires mypy and some other setup.

use a python virtual environment or conda to deal with systemwide pips from windows interefereing from running your ai applications
python -m will use a particular pythons pip, so you can download linux one and isolate it to linux pythons pip

pre cuda installation
https://docs.nvidia.com/cuda/archive/11.8.0/cuda-installation-guide-linux/index.html#pre-installation-actions

cuda installation
https://docs.nvidia.com/cuda/archive/11.8.0/cuda-installation-guide-linux/index.html#wsl-installation
https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local

post cuda installation
https://docs.nvidia.com/cuda/archive/11.8.0/cuda-installation-guide-linux/index.html#post-installation-actions

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