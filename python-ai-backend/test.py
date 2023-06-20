import numpy as np
from PIL import Image
from scipy.cluster.vq import kmeans

def extract_color_palette(image: Image.Image, num_colors: int = 256) -> list:
    # Convert the image to an array of RGB values
    image_array = np.array(image)

    # Normalize RGB values
    pixels = image_array.reshape(-1, 3) / 255.0

    # Perform k-means clustering to extract the dominant colors
    centroids, _ = kmeans(pixels, num_colors)

    # Create the color palette using the centroid values
    palette = (centroids * 255.0).astype(int).tolist()

    return palette



def generate_random_noise_image(size: tuple, palette: list, chunk_size: int) -> Image.Image:
    # Create a new image filled with random noise chunks using the color palette
    width, height = size
    num_colors = len(palette)

    # Calculate the number of chunks in each dimension
    num_chunks_x = width // chunk_size
    num_chunks_y = height // chunk_size

    # Generate random noise chunks
    random_chunks = np.random.choice(num_colors, size=(num_chunks_y, num_chunks_x))

    # Create an empty array for the final random noise image
    random_image_array = np.empty((height, width, 3), dtype=np.uint8)

    # Fill the random image array with random noise chunks
    for i in range(num_chunks_y):
        for j in range(num_chunks_x):
            chunk_color_index = random_chunks[i, j]
            chunk_color = palette[chunk_color_index]
            random_image_array[i * chunk_size: (i + 1) * chunk_size, j * chunk_size: (j + 1) * chunk_size] = chunk_color

    random_image = Image.fromarray(random_image_array, mode='RGB')

    return random_image

# Load the image
image_path = 'test.png'  # Replace with the actual path to your image
image = Image.open(image_path)

# Scale down the image
scaled_image = image.resize((128, 128))  # Adjust the size as per your preference

# Extract the color palette from the scaled image
palette = extract_color_palette(scaled_image, num_colors=256)

# Define the size of the random noise image
width, height = scaled_image.size
size = (1344, 1344)

# Generate the random noise image using the color palette
random_noise_image = generate_random_noise_image(size, palette,2)

# Display the original image, scaled image, and the random noise image
image.save("original.png")
scaled_image.save("scaled_orig.png")
random_noise_image.save("random_noise.png")