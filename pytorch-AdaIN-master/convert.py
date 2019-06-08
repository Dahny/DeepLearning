import os
from PIL import Image

directory = os.fsencode("input/content")

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    print(filename , directory)
    filepath = os.path.join(directory, filename)
    image = Image.open(filepath)
    print(image.mode)