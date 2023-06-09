import os
from PIL import Image
from utils.crypto import calculate_hash

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def save_file(image, file_name):
    file_hash = calculate_hash(file_name.replace(' ', '_').lower())
    filename = f"./gen_pics/{file_hash}.png"
    os.makedirs('gen_pics', exist_ok=True)  # Create the directory if it doesn't exist
    image.save(filename)
    return filename