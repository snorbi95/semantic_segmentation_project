import os
import glob
import cv2
import numpy as np
import tensorflow.python.keras.metrics
from matplotlib import pyplot as plt
from skimage import io, transform

label_codes, label_names = [(0,0,0),(64,64,64),(128,128,128),(192,192,192)], ['Background','Arthery','Ureter','Nerve']
#label_codes, label_names = [(0,0,0),(0,255,255)], ['Background','Arthery']
code2id = {v: k for k, v in enumerate(label_codes)}
id2code = {k: v for k, v in enumerate(label_codes)}

name2id = {v: k for k, v in enumerate(label_names)}
id2name = {k: v for k, v in enumerate(label_names)}
#
def rgb_to_onehot(rgb_image, colormap):
    '''Function to one hot encode RGB mask labels
        Inputs:
            rgb_image - image matrix (eg. 256 x 256 x 3 dimension numpy ndarray)
            colormap - dictionary of color to label id
        Output: One hot encoded image of dimensions (height x width x num_classes) where num_classes = len(colormap)
    '''
    num_classes = len(colormap)
    rgb_image = np.array(rgb_image)
    shape = rgb_image.shape[:2]+(num_classes,)
    encoded_image = np.zeros( shape, dtype=np.int8 )
    for i, cls in enumerate(colormap):
        encoded_image[:,:,i] = np.all(rgb_image.reshape( (-1,3) ) == colormap[i], axis=1).reshape(shape[:2])
    return encoded_image

def image_prepocessing(size_x, size_y,training_image_path, training_mask_path, test_image_path, test_mask_path, n_classes):
    #Resizing images, if needed
    SIZE_X = size_x
    SIZE_Y = size_y
    n_classes = n_classes #Number of classes for segmentation
    num_images = 653

    image_names = glob.glob(f"{training_image_path}/*.png") + glob.glob(f"{training_image_path}/*.jpg")
    image_names_test = glob.glob(f"{test_image_path}/*.png") + glob.glob(f"{test_image_path}/*.jpg")
    image_names.sort()
    image_names_test.sort()

    print('Load training images...')
    image_dataset = np.zeros((len(image_names), SIZE_X, SIZE_Y, 3))
    for i in range(len(image_names)):
        img = cv2.imread(image_names[i])
        img = cv2.resize(img, (SIZE_X, SIZE_Y))
        image_dataset[i] = img[:,:,:3]

    print('Load test images...')
    image_test_dataset = np.zeros((len(image_names_test), SIZE_X, SIZE_Y, 3))
    for i in range(len(image_names_test)):
        img = cv2.imread(image_names_test[i])
        img = cv2.resize(img, (SIZE_X, SIZE_Y))
        image_test_dataset[i] = img[:,:,:3]
    #image_dataset.reshape((len(image_dataset), SIZE_X, SIZE_X))
    #image_dataset = np.expand_dims(image_dataset, axis = 3)
    #image_test_dataset = np.expand_dims(image_test_dataset, axis = 3)


    mask_names = glob.glob(f"{training_mask_path}/*.png") + glob.glob(f"{training_mask_path}/*.jpg")
    mask_names_test = glob.glob(f"{test_mask_path}/*.png") + glob.glob(f"{test_mask_path}/*.jpg")
    mask_names.sort()
    mask_names_test.sort()

    print('Load training masks...')
    mask_dataset = np.zeros((len(mask_names), SIZE_X, SIZE_Y))
    mask_test_dataset = np.zeros((len(mask_names_test), SIZE_X, SIZE_Y))
    for i in range(len(mask_names)):
        img = cv2.imread(mask_names[i], 0)
        img = cv2.resize(img, (SIZE_X, SIZE_Y), interpolation=cv2.INTER_NEAREST)
        #img = rgb_to_onehot(img, id2code)
        mask_dataset[i] = img

    print('Load test masks...')
    for i in range(len(mask_names_test)):
        img = cv2.imread(mask_names_test[i], 0)
        img = cv2.resize(img, (SIZE_X, SIZE_Y), interpolation=cv2.INTER_NEAREST)
        #img = rgb_to_onehot(img, id2code)
        mask_test_dataset[i] = img

    print("Image data shape is: ", image_dataset.shape)
    print("Image test data shape is: ", image_test_dataset.shape)
    print("Mask data shape is: ", mask_dataset.shape)
    print("Mask test data shape is: ", mask_test_dataset.shape)
    print("Max pixel value in image is: ", image_dataset.max())
    print("Labels in the mask are : ", np.unique(mask_dataset))

    from sklearn.preprocessing import LabelEncoder
    labelencoder = LabelEncoder()
    n, h, w = mask_dataset.shape
    mask_dataset_reshaped = mask_dataset.reshape(-1,1)
    mask_dataset_reshaped_encoded = labelencoder.fit_transform(mask_dataset_reshaped)
    mask_dataset_encoded = mask_dataset_reshaped_encoded.reshape(n, h, w)

    n, h, w = mask_test_dataset.shape
    mask_test_dataset_reshaped = mask_test_dataset.reshape(-1,1)
    mask_test_dataset_reshaped_encoded = labelencoder.fit_transform(mask_test_dataset_reshaped)
    mask_test_dataset_encoded = mask_test_dataset_reshaped_encoded.reshape(n, h, w)

    # from sklearn.utils import class_weight
    # class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(mask_dataset_reshaped_encoded), y=mask_dataset_reshaped_encoded)
    # print("Class weights are...:", class_weights)

    print(np.unique(mask_dataset_encoded))


    mask_dataset_encoded = np.expand_dims(mask_dataset_encoded, axis = 3)
    mask_test_dataset_encoded = np.expand_dims(mask_test_dataset_encoded, axis = 3)
    print(mask_dataset_encoded.shape)


    image_dataset = image_dataset / 255
    image_test_dataset = image_test_dataset / 255

    X_train = image_dataset
    y_train = mask_dataset
    X_test = image_test_dataset
    y_test = mask_test_dataset

    from tensorflow.keras.utils import to_categorical
    import sklearn

    train_masks = to_categorical(y_train, num_classes=n_classes)
    y_train = train_masks.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))

    test_masks = to_categorical(y_test, num_classes=n_classes)
    y_test = test_masks.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))

    # num = np.random.randint(len(X_test))
    # fig, ax = plt.subplots(1,2)
    # ax[0].imshow(X_test[num])
    # ax[1].imshow(y_test[num,:,:])
    # plt.show()


    print(y_train.shape)
    from sklearn import model_selection
    X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train, y_train, test_size = 0.15)
    return X_train, y_train, X_val, y_val, X_test, y_test

import tensorflow as tf

def _read_to_tensor(fname, output_height=128, output_width=128, normalize_data=False):
    '''Function to read images from given image file path, and provide resized images as tensors
        Inputs:
            fname - image file path
            output_height - required output image height
            output_width - required output image width
            normalize_data - if True, normalize data to be centered around 0 (mean 0, range 0 to 1)
        Output: Processed image tensors
    '''

    # Read the image as a tensor
    img_strings = tf.io.read_file(fname)
    imgs_decoded = tf.image.decode_jpeg(img_strings)

    # Resize the image
    output = tf.image.resize(imgs_decoded, [output_height, output_width])

    # Normalize if required
    if normalize_data:
        output = (output - 128) / 128
    return output


def read_images(training_image_path, training_mask_path, test_image_path, test_mask_path):
    '''Function to get all image directories, read images and masks in separate tensors
        Inputs:
            img_dir - file directory
        Outputs
            frame_tensors, masks_tensors, frame files list, mask files list
    '''

    # Get the file names list from provided directory

    # Separate frame and mask files lists, exclude unnecessary files
    frames_list = [file for file in os.listdir(training_image_path)]
    masks_list = [file for file in os.listdir(training_mask_path)]

    frames_list.sort()
    masks_list.sort()

    print('{} frame files found in the provided directory.'.format(len(frames_list)))
    print('{} mask files found in the provided directory.'.format(len(masks_list)))

    # Create file paths from file names
    frames_paths = [os.path.join(training_image_path, fname) for fname in frames_list]
    masks_paths = [os.path.join(training_mask_path, fname) for fname in masks_list]

    # Create dataset of tensors
    frame_data = tf.data.Dataset.from_tensor_slices(frames_paths)
    masks_data = tf.data.Dataset.from_tensor_slices(masks_paths)

    # Read images into the tensor dataset
    frame_tensors = frame_data.map(_read_to_tensor)
    masks_tensors = masks_data.map(_read_to_tensor)

    print('Completed importing {} frame images from the provided directory.'.format(len(frames_list)))
    print('Completed importing {} mask images from the provided directory.'.format(len(masks_list)))

    return frame_tensors, masks_tensors, frames_list, masks_list

def rgb_to_onehot(rgb_image, colormap):
    '''Function to one hot encode RGB mask labels
        Inputs:
            rgb_image - image matrix (eg. 256 x 256 x 3 dimension numpy ndarray)
            colormap - dictionary of color to label id
        Output: One hot encoded image of dimensions (height x width x num_classes) where num_classes = len(colormap)
    '''
    num_classes = len(colormap)
    shape = rgb_image.shape[:2]+(num_classes,)
    encoded_image = np.zeros( shape, dtype=np.int8 )
    for i, cls in enumerate(colormap):
        encoded_image[:,:,i] = np.all(rgb_image.reshape( (-1,3) ) == colormap[i], axis=1).reshape(shape[:2])
    return encoded_image


def onehot_to_rgb(onehot, colormap):
    '''Function to decode encoded mask labels
        Inputs:
            onehot - one hot encoded image matrix (height x width x num_classes)
            colormap - dictionary of color to label id
        Output: Decoded RGB image (height x width x 3)
    '''
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros( onehot.shape[:2]+(3,) )
    for k in colormap.keys():
        output[single_layer==k] = colormap[k]
    return np.uint8(output)


seed = 1

"""### Custom image data generators for creating batches of frames and masks"""


def image_preprocessing_v2(size_x, size_y,training_image_path, training_mask_path, test_image_path, test_mask_path, n_classes):

    from keras.preprocessing.image import ImageDataGenerator
    frame_tensors, masks_tensors,frames_list, masks_list =  read_images(training_image_path, training_mask_path, test_image_path, test_mask_path)
    frame_batches = tf.compat.v1.data.make_one_shot_iterator(frame_tensors)
    mask_batches = tf.compat.v1.data.make_one_shot_iterator(masks_tensors)

    n_images_to_show = 5

    # for i in range(n_images_to_show):
    #     # Get the next image from iterator
    #     frame = frame_batches.next().numpy().astype(np.uint8)
    #     mask = mask_batches.next().numpy().astype(np.uint8)
    #
    #     # Plot the corresponding frames and masks
    #     fig = plt.figure()
    #     fig.add_subplot(1, 2, 1)
    #     plt.grid(b=None)
    #     plt.imshow(frame)
    #     fig.add_subplot(1, 2, 2)
    #     plt.grid(b=None)
    #     plt.imshow(mask)
    #     plt.show()
    label_codes, label_names = [(0,0,0),(0,255,0),(128,255,127),(255,255,0)], ['Background','Ureter','Arthery','Nerve']

    code2id = {v: k for k, v in enumerate(label_codes)}
    id2code = {k: v for k, v in enumerate(label_codes)}

    name2id = {v: k for k, v in enumerate(label_names)}
    id2name = {k: v for k, v in enumerate(label_names)}

    data_gen_args = dict(rescale=1. / 255)
    mask_gen_args = dict()

    train_frames_datagen = ImageDataGenerator(**data_gen_args)
    train_masks_datagen = ImageDataGenerator(**mask_gen_args)
    val_frames_datagen = ImageDataGenerator(**data_gen_args)
    val_masks_datagen = ImageDataGenerator(**mask_gen_args)

    def TrainAugmentGenerator(seed=1, batch_size=5):
        '''Train Image data generator
            Inputs:
                seed - seed provided to the flow_from_directory function to ensure aligned data flow
                batch_size - number of images to import at a time
            Output: Decoded RGB image (height x width x 3)
        '''
        train_image_generator = train_frames_datagen.flow_from_directory(
            training_image_path,
            batch_size=batch_size, seed=seed, target_size=(size_x, size_y))

        train_mask_generator = train_masks_datagen.flow_from_directory(
            training_mask_path,
            batch_size=batch_size, seed=seed, target_size=(size_x, size_y))

        while True:
            X1i = train_image_generator.next()
            X2i = train_mask_generator.next()

            # One hot encoding RGB images
            mask_encoded = [rgb_to_onehot(X2i[0][x, :, :, :], id2code) for x in range(X2i[0].shape[0])]

            yield X1i[0], np.asarray(mask_encoded)


    def ValAugmentGenerator(seed=1, batch_size=5):
        '''Validation Image data generator
            Inputs:
                seed - seed provided to the flow_from_directory function to ensure aligned data flow
                batch_size - number of images to import at a time
            Output: Decoded RGB image (height x width x 3)
        '''
        val_image_generator = val_frames_datagen.flow_from_directory(
            test_image_path,
            batch_size=batch_size, seed=seed, target_size=(size_x, size_y))


        val_mask_generator = val_masks_datagen.flow_from_directory(
            test_mask_path,
            batch_size=batch_size, seed=seed, target_size=(size_x, size_y))
        while True:
            X1i = val_image_generator.next()
            X2i = val_mask_generator.next()

            # One hot encoding RGB images
            mask_encoded = [rgb_to_onehot(X2i[0][x, :, :, :], id2code) for x in range(X2i[0].shape[0])]

            yield X1i[0], np.asarray(mask_encoded)

    train_generator = TrainAugmentGenerator()
    val_generator = ValAugmentGenerator()
    return train_generator, val_generator

# image_preprocessing_v2(192,192,'dataset/train/images', 'dataset/train/mask', 'dataset/test/images', 'dataset/test/mask', 4)
