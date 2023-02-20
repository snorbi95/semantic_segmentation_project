from skimage import io, color, filters
import matplotlib.pyplot as plt
import numpy as np

image = io.imread(f'res_input.png')



# hsv_image = color.rgb2hsv(image)
# # fig, ax = plt.subplots(1,3)
# # ax[0].imshow(hsv_image[:, :, 0])
# # ax[1].imshow(hsv_image[:, :, 1])
# # ax[2].imshow(hsv_image[:, :, 2])
# # plt.show()
# threshold = filters.threshold_mean(hsv_image[:, :, 0])
# gray_image = hsv_image[:, :, 2]
# gray_image[hsv_image[:, :, 0] > threshold - 0.15] = 0
# plt.imsave(f'gray.jpg',gray_image, cmap = 'gray')
# # plt.imshow((gray_image_2 + gray_image) / 2)
# # plt.show()
# edge_image = filters.sobel(gray_image)
# # image = (color.rgb2gray(image) * 255).astype(np.uint8)
# plt.imsave(f'edge.jpg',edge_image, cmap = 'gray')