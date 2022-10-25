from skimage import io
import matplotlib.pyplot as plt

mask = io.imread(f'../dataset/test_dataset_crop/mask/movie_2.mov_0.036667.png.png_0.png')

plt.imshow(mask)
plt.show()