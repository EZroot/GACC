#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

# Specify the Miniconda Python executable path
miniconda_path="/path/to/python"

# Run the Python script using the Miniconda Python executable
"${miniconda_path}" ./stablediffusion/stable_diffusion_server.py
