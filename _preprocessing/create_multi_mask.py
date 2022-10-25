import shutil

import numpy as np
from skimage import io, color
import os
import matplotlib.pyplot as plt


image_path = f'../images'
mask_path = f'../mask'


image_names = [f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f)) and os.path.join(image_path, f).find('.ini') == -1]
mask_names = [f for f in os.listdir(mask_path) if os.path.isfile(os.path.join(mask_path, f))]

#RGB
# label_values = {'arthery': 0, 'ureter': 128, 'nerve': 255}
# #label_values = {'arthery': 0, 'ureter': 0, 'nerve': 0}
# for image_name in image_names:
#     print(image_name)
#     image = io.imread(f'{image_path}/{image_name}')
#     image_name = image_name.split('.png')[0]
#     multi_mask = np.zeros((image.shape[0], image.shape[1], 3)).astype(np.uint8)
#     print(multi_mask.shape)
#     for mask_name in mask_names:
#         if image_name in mask_name:
#             for key, label in label_values.items():
#                 mask_name_splitted = mask_name.split('_')[-1]
#                 if key in mask_name_splitted:
#                     mask_image = io.imread(f'{mask_path}/{mask_name}')
#                     multi_mask[mask_image != 0] = [label, 255, 255 - label]
#                     # plt.imshow(multi_mask)
#                     # plt.show()
#     io.imsave(f'multi_mask/visible/{image_name}.png.png', multi_mask)

#Grayscale
label_values = {'arthery': 1, 'ureter': 2, 'nerve': 3}
#
for image_name in image_names:
    print(image_name)
    image = io.imread(f'{image_path}/{image_name}')
    image_name = image_name.split('.png')[0]
    multi_mask = np.zeros((image.shape[0], image.shape[1])).astype(np.uint8)
    print(multi_mask.shape)
    for mask_name in mask_names:
        if image_name in mask_name:
            for key, label in label_values.items():
                mask_name_splitted = mask_name.split('_')[-1]
                if key in mask_name_splitted:
                    mask_image = io.imread(f'{mask_path}/{mask_name}')
                    multi_mask[mask_image != 0] = label
                    # plt.imshow(multi_mask
                    # plt.show()
    io.imsave(f'../multi_mask/{image_name}.png.png', multi_mask)

arthery_mask_path = '../dataset/train/mask'
arthery_mask_names = os.listdir(arthery_mask_path)
arthery_image_path = '../dataset/train/images'
arthery_image_names = os.listdir(arthery_image_path)

# for image_name in arthery_image_names:
#     if image_name not in arthery_mask_names:
#         print(image_name)


# #only arthery
# for mask_name in arthery_mask_names:
#     mask_name_cropped = mask_name.split('.png')[0]
#     #print(mask_name)
#     for image_name in image_names:
#         image_name_cropped = image_name.split('.png')[0]
#         if mask_name_cropped == image_name_cropped:
#             mask_image = io.imread(f'{arthery_mask_path}/{mask_name}')
#             mask_image[mask_image != 0] = 1
#             io.imsave(f'{arthery_mask_path}/{mask_name}', mask_image)
            #shutil.copyfile(f'{image_path}/{image_name}', f'{arthery_image_path}/{image_name}')

# extended_mask_path = 'preview_arthery/mask'
# mask_names = os.listdir(extended_mask_path)
#
# for image_name in mask_names:
#     mask_image = io.imread(f'{extended_mask_path}/{image_name}')
#     mask_image[mask_image != 0] = 1
#     io.imsave(f'{extended_mask_path}/{image_name}', mask_image)