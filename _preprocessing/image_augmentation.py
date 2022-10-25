import os

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=80,
        zoom_range=0.4,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

image_path = '../arthery_dataset/train/images'
mask_path = '../arthery_dataset/train/mask'
image_list = [f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path,f))]
mask_list = [f for f in os.listdir(mask_path) if os.path.isfile(os.path.join(mask_path,f))]
seed = 1
for image_idx in range(len(image_list)):
    print(f'Augmenting {image_list[image_idx]}')
    img = load_img(os.path.join(image_path, image_list[image_idx]))  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

    img = load_img(os.path.join(mask_path, mask_list[image_idx]))  # this is a PIL image
    y = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    y = y.reshape((1,) + y.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `preview/` directory
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir='../preview_arthery/images', save_prefix = image_list[image_idx], save_format='png', seed=seed):
        i += 1
        if i > 3:
            break  # otherwise the generator would loop indefinitely

    i = 0
    for batch in datagen.flow(y, batch_size=1,
                              save_to_dir='../preview_arthery/mask', save_prefix = mask_list[image_idx], save_format='png', seed=seed):
        i += 1
        if i > 3:
            break  # otherwise the generator would loop indefinitely
