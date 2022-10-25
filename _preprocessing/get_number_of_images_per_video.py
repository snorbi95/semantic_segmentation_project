import os

import cv2
import numpy as np
from skimage import io

video_num_dict = {}

train_image_names = os.listdir(f'../dataset/test_dataset/images')
test_image_names = os.listdir(f'../ureter_dataset/test/images')
negative_image_train_names = os.listdir(f'../ureter_dataset/negative_images/train')
negative_image_test_names = os.listdir(f'../ureter_dataset/negative_images/test')

img_size = (512,512)
name_list = [train_image_names, test_image_names, negative_image_train_names, negative_image_test_names]

for lst in name_list:
    for image_name in lst:
        title = image_name.split('.')[0]
        movie_num = int(title.split('_')[1])
        if movie_num not in video_num_dict:
            video_num_dict[movie_num] = 1
        else:
            video_num_dict[movie_num] += 1

for k, v in video_num_dict.items():
    video_num_image_names = [image_name for image_name in train_image_names if int(image_name.split('.')[0].split('_')[1]) == k]
    image_path = f'../dataset/test_dataset/images'
    mask_path = f'../dataset/test_dataset/mask'
    if len(video_num_image_names) != 0:
        slices_per_image = 60 // len(video_num_image_names)
        last_slices_plus = 60 % len(video_num_image_names)
        for i, video_num_image_name in enumerate(video_num_image_names):
            if i == len(video_num_image_names) - 1:
                current_number_of_slices = slices_per_image + last_slices_plus
            else:
                current_number_of_slices = slices_per_image
            full_image_path = os.path.join(image_path, video_num_image_name)
            full_image = cv2.imread(full_image_path, cv2.IMREAD_COLOR)
            image_name = video_num_image_name.replace('.jpg', '.png')
            full_mask = cv2.imread(os.path.join(mask_path, image_name), cv2.IMREAD_GRAYSCALE)
            #full_mask[full_mask != 0] = 1
            full_mask_len = np.sum(full_mask)
            # mask_values = str(np.unique(full_mask))
            if full_mask_len != 0:
                # if mask_values == '[0 1]':
                #     slices = 24
                # elif mask_values == '[0 2]':
                #     slices = 16
                # elif mask_values == '[0 3]':
                #     slices = 26
                # elif mask_values == '[0 1 2]':
                #     slices = 98
                for i in range(current_number_of_slices):
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
                    io.imsave(f'../dataset/test_dataset_crop/images/{image_name}_{i}.png', image)
                    io.imsave(f'../dataset/test_dataset_crop/mask/{image_name}_{i}.png', mask)


# print(len(list(video_num_dict.keys())))
# print(np.average(list(video_num_dict.values())))
# print(np.median(list(video_num_dict.values())))

