import numpy as np
from stl import mesh
from PIL import Image

# Define the 8 vertices of the cube
vertices = np.array([
    [-1, -1, -1],
    [+1, -1, -1],
    [+1, +1, -1],
    [-1, +1, -1]])

# Define the 12 triangles composing the cube
faces = np.array([
    [1, 2, 3],
    [3, 1, 0]
])

# Create the mesh
cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces):
    for j in range(3):
        cube.vectors[i][j] = vertices[f[j], :]

# Write the mesh to file "cube.stl"
cube.save('surface.stl')

# Load and convert the image to grayscale
grey_img = Image.open('as.png').convert('L')

# Set the maximum size of the image texture
max_size = (512, 512)
max_height = 10
# Resize the image while maintaining aspect ratio
grey_img.thumbnail(max_size)
imageNp = np.array(grey_img)
maxPix = imageNp.max()
minPix = imageNp.min()

(ncols, nrows) = grey_img.size

vertices = np.zeros((nrows, ncols, 3))
uv_coords = np.zeros((nrows, ncols, 2))

for x in range(ncols):
    for y in range(nrows):
        pixelIntensity = imageNp[y][x]
        z = (pixelIntensity * max_height) / maxPix

        # Calculate UV coordinates
        u = x / (ncols - 1)
        v = y / (nrows - 1)

        vertices[y][x] = (x, y, z)
        uv_coords[y][x] = (u, v)

faces = []

for x in range(ncols - 1):
    for y in range(nrows - 1):
        # Create face 1
        vertice1 = vertices[y][x]
        vertice2 = vertices[y + 1][x]
        vertice3 = vertices[y + 1][x + 1]
        uv1 = uv_coords[y][x]
        uv2 = uv_coords[y + 1][x]
        uv3 = uv_coords[y + 1][x + 1]
        face1 = np.array([(vertice1, uv1), (vertice2, uv2), (vertice3, uv3)])

        # Create face 2
        vertice1 = vertices[y][x]
        vertice2 = vertices[y][x + 1]
        vertice3 = vertices[y + 1][x + 1]
        uv1 = uv_coords[y][x]
        uv2 = uv_coords[y][x + 1]
        uv3 = uv_coords[y + 1][x + 1]
        face2 = np.array([(vertice1, uv1), (vertice2, uv2), (vertice3, uv3)])

        faces.append(face1)
        faces.append(face2)

print(f"Number of faces: {len(faces)}")
facesNp = np.array(faces)

# Create the mesh
surface = mesh.Mesh(np.zeros(facesNp.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces):
    for j in range(3):
        surface.vectors[i][j] = facesNp[i][j][0]

# Set the texture coordinates
surface.uv = facesNp[:, :, 1]

# Write the mesh to file "surface.stl"
surface.save('surface.stl')
print(surface)
