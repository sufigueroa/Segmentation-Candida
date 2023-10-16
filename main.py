import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from utils import *

PATH = 'images/Low concentration 1.tiff'
# Leer tiff y convertirlo en np.array
im = read_tiff(PATH)

# Interpolacion isotropica
im = isotropic_interpolation(im)

for i in range(len(im)):
    # save_image(im[i], f'results/non_seg_{i}.png')
    segmentated = segmentation(im[i])
    save_image(segmentated, f'results/seg_{i}.png')