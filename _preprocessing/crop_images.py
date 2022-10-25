import os, re
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import io
from sklearn.utils import class_weight

mode = 'train'
# image_path = f'../ureter_dataset/negative_images/{mode}'
# mask_path = f'../ureter_dataset/negative_masks/{mode}'
#image_path = f'../dataset/test_dataset/images_nerve'
image_path = f'../dataset/for_figure/image'
#mask_path = f'../dataset/test_dataset/mask_nerve'
mask_path = f'../dataset/for_figure/mask'
img_size = (512,512)
slices = 24

image_names = [f for f in os.listdir(image_path)]
values_dict = {'0': 0, '1': 1}

for image_name in image_names:
    full_image = cv2.imread(os.path.join(image_path, image_name), cv2.IMREAD_COLOR)
    image_name = image_name.replace('.jpg','.png')
    full_mask = cv2.imread(os.path.join(mask_path, image_name), cv2.IMREAD_GRAYSCALE)
    #full_mask[full_mask != 0] = 1
    # values_dict['0'] += full_mask[full_mask == 0].size
    # values_dict['1'] += full_mask[full_mask != 0].size
    # print(values_dict)
    full_mask_len = np.sum(full_mask)
    #mask_values = str(np.unique(full_mask))
    if full_mask_len != 0:
        # if mask_values == '[0 1]':
        #     slices = 24
        # elif mask_values == '[0 2]':
        #     slices = 16
        # elif mask_values == '[0 3]':
        #     slices = 26
        # elif mask_values == '[0 1 2]':
        #     slices = 98
        for i in range(slices):
            random_x = np.random.randint(0, full_image.shape[0] - img_size[0])
            random_y = np.random.randint(0, full_image.shape[1] - img_size[1])
            mask = full_mask[random_x:random_x + img_size[0], random_y:random_y + img_size[1]]
            while np.sum(mask) < full_mask_len * 0.35:
                random_x = np.random.randint(0, full_image.shape[0] - img_size[0])
                random_y = np.random.randint(0, full_image.shape[1] - img_size[1])
                mask = full_mask[random_x:random_x + img_size[0], random_y:random_y + img_size[1]]
            #save_mask = np.zeros((img_size[0], img_size[1]))
            # plt.imshow(mask)
            # plt.show()
            # image = cv2.resize(image, (img_size[0], img_size[1]), interpolation=cv2.INTER_NEAREST)
            image = full_image[random_x:random_x + img_size[0], random_y:random_y + img_size[1]]
            # mask = cv2.resize(mask, (img_size[0], img_size[1]), interpolation=cv2.INTER_NEAREST)
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            print(f'Saving {i}th slice of image: {image_name}')
            # io.imsave(f'../ureter_dataset/{mode}_crop_w_negative/images/{image_name}_{i}.png', image)
            # io.imsave(f'../ureter_dataset/{mode}_crop_w_negative/mask/{image_name}_{i}.png', mask)
            #io.imsave(f'../dataset/test_dataset_crop/images/{image_name}_{i}.png', image)
            io.imsave(f'../dataset/for_figure/slices/{image_name}_{i}.png', image)
            #io.imsave(f'../dataset/test_dataset_crop/mask/{image_name}_{i}.png', mask)
# weight_0 = (values_dict['0']) / ((values_dict['0']) + (values_dict['1']))
# weight_1 = (values_dict['1']) / ((values_dict['0']) + (values_dict['1']))
# print(f'weights: { 1 - weight_0,1 - weight_1}')