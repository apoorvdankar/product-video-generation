import imageio
from PIL import Image
import numpy as np

im1 = Image.open("gen_image.jpg")
im2 = Image.open("gen1.jpg")

imageio.mimsave("aq.mp4", [np.array(im1), np.array(im2)], format="mp4", fps=2)
