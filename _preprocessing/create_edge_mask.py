import os
from skimage import io
from skimage.feature import canny
import matplotlib.pyplot as plt
import numpy as np
from skimage import morphology

file_path = f'../dataset/test/mask/movie_6.mov_1.332000.png.png'
image = io.imread(file_path)
image = (255 // image).astype(np.uint8)
image = canny(image)
image = morphology.dilation(image, selem = morphology.disk(2))

io.imsave(f'../dataset/test/edge_image.png', image)

