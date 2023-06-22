import numpy as np
from PIL import Image, ImageFilter
from scipy.cluster.vq import kmeans
from joblib import Parallel, delayed
import os
import random
from scipy.ndimage.filters import gaussian_filter

def generate_random_noise_image(size: tuple, palette: list, chunk_size: int, blur_radius: float) -> Image.Image:
    # Create a new image filled with random noise chunks using the color palette
    width, height = size
    num_colors = len(palette)

    # Calculate the number of chunks in each dimension
    num_chunks_x = (width + chunk_size - 1) // chunk_size  # Round up the division
    num_chunks_y = (height + chunk_size - 1) // chunk_size  # Round up the division

    # Generate random noise chunks with a Gaussian distribution
    mean = num_colors // 2  # Mean value for Gaussian distribution
    std = num_colors // 4  # Standard deviation for Gaussian distribution
    random_chunks = np.random.normal(mean, std, size=(num_chunks_y, num_chunks_x)).astype(int)

    # Create an empty array for the final random noise image
    random_image_array = np.empty((height, width, 3), dtype=np.uint8)

    # Fill the random image array with random noise chunks
    for i in range(num_chunks_y):
        for j in range(num_chunks_x):
            chunk_color_index = random_chunks[i, j] % num_colors
            chunk_color = palette[chunk_color_index]
            x_start = j * chunk_size
            y_start = i * chunk_size
            x_end = min(x_start + chunk_size, width)
            y_end = min(y_start + chunk_size, height)
            random_image_array[y_start:y_end, x_start:x_end] = chunk_color

    # Create a copy of the random image array for blurring
    blurred_image_array = np.copy(random_image_array)

    # Apply Gaussian blur to each color channel separately
    for c in range(3):
        blurred_image_array[:, :, c] = gaussian_filter(blurred_image_array[:, :, c], sigma=blur_radius)

    blurred_image = Image.fromarray(blurred_image_array, mode='RGB')

    return blurred_image

def extract_color_palette(image: Image.Image, num_colors: int = 256) -> list:
    # Convert the image to an array of RGB values
    image_array = np.array(image)

    # Normalize RGB values
    pixels = image_array.reshape(-1, 3) / 255.0

    def compute_kmeans(pixels, num_colors):
        # Perform k-means clustering to extract the dominant colors
        centroids, _ = kmeans(pixels, num_colors)
        return (centroids * 255.0).astype(int).tolist()

    # Split the pixels into chunks for parallel processing
    num_pixels = pixels.shape[0]
    chunk_size = num_pixels // num_colors
    pixel_chunks = [pixels[i:i+chunk_size] for i in range(0, num_pixels, chunk_size)]

    # Determine the number of threads based on the image size
    num_threads = os.cpu_count()
    if num_threads is None or num_threads < 1:
        num_threads = 1

    # Run k-means clustering in parallel
    results = Parallel(n_jobs=num_threads)(delayed(compute_kmeans)(chunk, num_colors) for chunk in pixel_chunks)

    # Concatenate the results from all chunks
    palette = np.concatenate(results).tolist()

    return palette

def upscale_and_paste(image: Image.Image, original_image: Image.Image, blur_radius: float) -> Image.Image:
    # Upscale the image to twice its size
    upscaled_image = image.resize((image.width * 2, image.height * 2), Image.BICUBIC)

    # Calculate the coordinates to paste the original image in the center
    paste_x = (upscaled_image.width - original_image.width) // 2
    paste_y = (upscaled_image.height - original_image.height) // 2

    # Create a blurred version of the original image
    blurred_image = original_image.filter(ImageFilter.GaussianBlur(blur_radius))

    # Create edge masks for the top, bottom, left, and right edges
    edge_mask = Image.new("L", original_image.size, 0)
    edge_width = int(original_image.width * 0.1)  # Adjust the width ratio as desired
    edge_height = int(original_image.height * 0.1)  # Adjust the height ratio as desired
    edge_mask.paste(255, (0, 0, original_image.width, edge_height))  # Top edge
    edge_mask.paste(255, (0, original_image.height - edge_height, original_image.width, original_image.height))  # Bottom edge
    edge_mask.paste(255, (0, 0, edge_width, original_image.height))  # Left edge
    edge_mask.paste(255, (original_image.width - edge_width, 0, original_image.width, original_image.height))  # Right edge

    # Apply Gaussian blur to the edge masks
    blurred_edge_mask = edge_mask.filter(ImageFilter.GaussianBlur(blur_radius))

    # Invert the blurred edge masks
    inverted_blurred_edge_mask = Image.eval(blurred_edge_mask, lambda x: 255 - x)

    # Create a new image to hold the final result
    final_image = Image.new("RGB", upscaled_image.size)

    # Paste the upscaled image onto the final image
    final_image.paste(upscaled_image, (0, 0))

    # Paste the blurred image in the center
    final_image.paste(original_image, (paste_x, paste_y), mask=inverted_blurred_edge_mask)

    return final_image


# Example usage
image_path = "test.png"
image = Image.open(image_path)

# Extract the color palette
palette = extract_color_palette(image)

# Generate random noise image using the extracted palette
size = image.size
chunk_size = 50
blur_radius = 10
blur_radius_edge = 6
random_image = generate_random_noise_image(size, palette, chunk_size,blur_radius)

# Display the random noise image
random_image.save("randomized.png")

# Upscale and paste the original image
upscaled_and_pasted_image = upscale_and_paste(random_image, image, blur_radius_edge)

# Save the final image
upscaled_and_pasted_image.save("final_image.png")

# import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense, Reshape
# from keras.optimizers import Adam
# import matplotlib.pyplot as plt
# import cv2

# # Set the dimensions of the latent noise vector
# latent_dim = 100

# # Build the generator model
# generator = Sequential()
# generator.add(Dense(128, input_dim=latent_dim, activation='relu'))
# generator.add(Dense(784, activation='sigmoid'))
# generator.add(Reshape((28, 28)))

# # Compile the generator model
# generator.compile(loss='binary_crossentropy', optimizer=Adam())

# # Generate random latent noise vectors
# num_samples = 1  # Number of noise samples to generate
# latent_noise = np.random.normal(0, 1, (num_samples, latent_dim))

# # Generate images from the latent noise
# generated_images = generator.predict(latent_noise)

# # Convert the image to the range [0, 255]
# generated_image = (generated_images[0] * 255).astype(np.uint8)

# # Save the generated image
# cv2.imwrite("generated_image.png", generated_image)
#------------------------------------------------------------------------------------------
# import numpy as np
# from PIL import Image
# from scipy.cluster.vq import kmeans

# def extract_color_palette(image: Image.Image, num_colors: int = 256) -> list:
#     # Convert the image to an array of RGB values
#     image_array = np.array(image)

#     # Normalize RGB values
#     pixels = image_array.reshape(-1, 3) / 255.0

#     # Perform k-means clustering to extract the dominant colors
#     centroids, _ = kmeans(pixels, num_colors)

#     # Create the color palette using the centroid values
#     palette = (centroids * 255.0).astype(int).tolist()

#     return palette



# def generate_random_noise_image(size: tuple, palette: list, chunk_size: int) -> Image.Image:
#     # Create a new image filled with random noise chunks using the color palette
#     width, height = size
#     num_colors = len(palette)

#     # Calculate the number of chunks in each dimension
#     num_chunks_x = width // chunk_size
#     num_chunks_y = height // chunk_size

#     # Generate random noise chunks
#     random_chunks = np.random.choice(num_colors, size=(num_chunks_y, num_chunks_x))

#     # Create an empty array for the final random noise image
#     random_image_array = np.empty((height, width, 3), dtype=np.uint8)

#     # Fill the random image array with random noise chunks
#     for i in range(num_chunks_y):
#         for j in range(num_chunks_x):
#             chunk_color_index = random_chunks[i, j]
#             chunk_color = palette[chunk_color_index]
#             random_image_array[i * chunk_size: (i + 1) * chunk_size, j * chunk_size: (j + 1) * chunk_size] = chunk_color

#     random_image = Image.fromarray(random_image_array, mode='RGB')

#     return random_image

# # Load the image
# image_path = 'test.png'  # Replace with the actual path to your image
# image = Image.open(image_path)

# # Scale down the image
# scaled_image = image.resize((128, 128))  # Adjust the size as per your preference

# # Extract the color palette from the scaled image
# palette = extract_color_palette(scaled_image, num_colors=256)

# # Define the size of the random noise image
# width, height = scaled_image.size
# size = (1344, 1344)

# # Generate the random noise image using the color palette
# random_noise_image = generate_random_noise_image(size, palette,2)

# # Display the original image, scaled image, and the random noise image
# image.save("original.png")
# scaled_image.save("scaled_orig.png")
# random_noise_image.save("random_noise.png")