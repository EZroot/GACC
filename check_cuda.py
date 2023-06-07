import torch
import tensorflow as tf

print(tf.config.list_physical_devices('GPU'))
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA device count:", torch.cuda.device_count())
    print("Current CUDA device:", torch.cuda.current_device())
    print("CUDA device name:", torch.cuda.get_device_name(device))
else:
    print("CUDA is not available.")
