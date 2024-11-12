from PIL import Image

image_path = 'assets/gettyimages-1367336961-640x640.jpg'
image = Image.open(image_path).convert('RGB')
width, height = image.size

filtered_image = Image.new('RGB', (width, height), 'black')
pixels = image.load()
filtered_pixels = filtered_image.load()

for x in range(width):
    for y in range(height):
        r, g, b = pixels[x, y]

        # Filter based on color ranges
        if 2 * r - g + b - 300 > 0:
            filtered_pixels[x, y] = (r, g, b)

filtered_image.save('results/filtered_image.jpg')
