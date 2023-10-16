from numpy import cos, sin
import numpy as np
import math
from PIL import Image
import cv2
from skspatial.objects import Line, Plane, Vector
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator


###################################################
### Funciones Generales
###################################################

# https://stackoverflow.com/questions/18602525/python-pil-for-loop-to-work-with-multi-image-tiff
# Funcion para abrir la imagen tiff y dejarla como una matriz
def read_tiff(path):
    """
    path - Path to the multipage-tiff file
    """
    img = Image.open(path)
    images = []
    for i in range(img.n_frames):
        img.seek(i)
        images.append(np.array(img))
    return np.array(images)

def save_image(img, path):
    img = Image.fromarray(np.uint8(img), 'L')
    img.save(path)

def save_video(frames, path):
    video = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'XVID'), 10, (frames[0].shape[1], frames[0].shape[0]), 0)
    for frame in frames:
        video.write(np.uint8(frame))
    video.release()

###################################################


###################################################
### Funciones de Interpolacion
###################################################

def isotropic_interpolation(matrix):
    new = []
    for i in range(len(matrix) - 1):
        img1 = matrix[i]
        img2 = matrix[i + 1]
        ab = np.dstack((img1, img2))
        inter = np.mean(ab, axis=2, dtype=ab.dtype) 
        new.append(img1)
        new.append(inter)
    new.append(img2)
    return np.array(new)

###################################################



def equalize(matrix):
    max_value = np.max(matrix)
    return matrix / max_value * 255

def initial_segmentation(img, umbral):
    segmented = (img > umbral) * 255
    return segmented.astype(np.uint8)

def denoise(img):
    # k_erosion = np.ones((6, 6), np.uint8) 
    # k_dilation = np.ones((9,9), np.uint8) 
    # img_dilation = cv2.dilate(img, k_dilation) 
    # img_erosion = cv2.erode(img_dilation, k_erosion)
    k_erosion = np.ones((3,3), np.uint8) 
    k_dilation = np.ones((15,15), np.uint8) 
    img_dilation = cv2.dilate(img, k_erosion) 
    img_erosion = cv2.erode(img_dilation, k_erosion)
    img_dilation = cv2.dilate(img_erosion, k_dilation) 
    img_erosion = cv2.erode(img_dilation, np.ones((9,9), np.uint8) )

    # k_erosion = np.ones((3,3), np.uint8) 
    # k_dilation = np.ones((9,9), np.uint8) 
    # img_erosion = cv2.erode(img, k_erosion)
    # img_dilation = cv2.dilate(img_erosion, k_dilation) 
    return img_erosion

def segmentation(img):
    segmented = initial_segmentation(img, 40)
    denoised = denoise(segmented)
    return denoised

def histograa():
    # flat = im[16].flatten()

    # flat = flat / flat.max() * 255

    # # plot:
    # fig, ax = plt.subplots()

    # ax.hist(flat, bins=100, edgecolor="white")

    # # ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
    # #        ylim=(0, 56), yticks=np.linspace(0, 56, 9))

    # plt.show()
    pass