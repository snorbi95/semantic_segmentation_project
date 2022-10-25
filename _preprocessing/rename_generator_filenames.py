import os

image_path = f'../dataset/test_gen/mask/mask'

image_names = [f for f in os.listdir(image_path)]

n = 1

for image_name in image_names:
    image_name_new = f'{n}.png'
    os.rename(f'{image_path}/{image_name}', f'{image_path}/{image_name_new}')
    n += 1