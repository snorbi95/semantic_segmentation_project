import os
import numpy as np
import shutil

test = [21,22]
valid = []

# image_path = f'images'
# image_names = [f for f in os.listdir(image_path) if os.path.join(image_path, f).find('desktop') == -1]
#
# for image_name in image_names:
#     title = image_name.split('.')[0]
#     movie_num = int(title.split('_')[1])
#     if movie_num not in test:
#         shutil.copyfile(f'{image_path}/{image_name}', f'dataset/train/images/{image_name}')

mask_image_path = f'../ureter_dataset/all_masks'
rgb_image_path = f'../ureter_dataset/all_images'
# image_path = f'arthery_dataset/arthery_mask'
# rgb_image_path = f'arthery_dataset/arthery_images'

mask_image_names = [f for f in os.listdir(mask_image_path) if os.path.join(mask_image_path, f).find('desktop') == -1 and os.path.isfile(os.path.join(mask_image_path, f))]
rgb_image_names = [f for f in os.listdir(rgb_image_path) if os.path.join(rgb_image_path, f).find('desktop') == -1 and os.path.isfile(os.path.join(rgb_image_path, f))]

for image_name in mask_image_names:
    title = image_name.split('.')[0]
    movie_num = int(title.split('_')[1])
    if movie_num not in test:
        pass
        # shutil.copyfile(f'{mask_image_path}/{image_name}', f'../nerve_dataset/train_crop/mask/{image_name}')
        # shutil.copyfile(f'{rgb_image_path}/{image_name}', f'../nerve_dataset/train_crop/images/{image_name}')
        #shutil.copyfile(f'{image_path}/{image_name}', f'arthery_dataset/train/mask/{image_name}')
    else:
        shutil.copyfile(f'{mask_image_path}/{image_name}', f'../ureter_dataset/test_cv_set/mask/{image_name}')
        shutil.copyfile(f'{rgb_image_path}/{image_name}', f'../ureter_dataset/test_cv_set/images/{image_name}')
        #shutil.copyfile(f'{image_path}/{image_name}', f'arthery_dataset/test/mask/{image_name}')

# for image_name in rgb_image_names:
#     title = image_name.split('.')[0]
#     movie_num = int(title.split('_')[1])
#     if movie_num not in test:
#         # shutil.copyfile(f'{image_path}/{image_name}', f'dataset/train/mask/{image_name}')
#         #shutil.copyfile(f'{rgb_image_path}/{image_name}', f'dataset/train/images/{image_name}')
#         shutil.copyfile(f'{rgb_image_path}/{image_name}', f'arthery_dataset/train/images/{image_name}')
#     else:
#         #shutil.copyfile(f'{rgb_image_path}/{image_name}', f'dataset/test/images/{image_name}')
#         # shutil.copyfile(f'{image_path}/{image_name}', f'dataset/test/mask/{image_name}')
#         shutil.copyfile(f'{rgb_image_path}/{image_name}', f'arthery_dataset/test/images/{image_name}')