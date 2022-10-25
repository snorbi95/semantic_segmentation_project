import os, re

image_path = f'../nerve_dataset/nerve_masks'

image_names = [f for f in os.listdir(image_path) if os.path.join(image_path, f).find('desktop') == -1]

name_dict = {}
movie_count = 1

for image_name in image_names:
    # title = image_name.split('.')[0]
    # if title not in name_dict:
    #     name_dict[title] = f'movie_{movie_count}'
    #     movie_count += 1
    image_name_new = image_name.replace('_nerve', '')
    os.rename(f'{image_path}/{image_name}', f'{image_path}/{image_name_new}')
