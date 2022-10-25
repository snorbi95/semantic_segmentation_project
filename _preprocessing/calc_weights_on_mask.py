# import os, re
# from skimage import io
# import matplotlib.pyplot as plt
import numpy as np
#
# image_path = f'../dataset/train_crop/mask'
#
# image_names = [f for f in os.listdir(image_path) if os.path.join(image_path, f).find('desktop') == -1]
# pix_sum = np.zeros(4)
#
# for image_name in image_names:
#     image = io.imread(f'{image_path}/{image_name}')
#     for i in range(4):
#         pix_sum[i] += image[image == i].size
#
# pix_num = len(image_names) * (512 * 512)
#weights = pix_sum / pix_num
weights = np.array([0.91607619, 0.02096508, 0.0591985,  0.00376023])
weights = 1 - weights
print(weights)