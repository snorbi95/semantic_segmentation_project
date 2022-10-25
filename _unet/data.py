import os
import numpy as np
import pandas as pd
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
import cv2
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img

num_classes = 4

class Data(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)
            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
            #y[j] -= 1
        return x, y

def process_data(data_path, file_path):

    images = [os.path.join(os.path.join(f'{data_path}/{file_path}', 'images', f)) for f in os.listdir(os.path.join(f'{data_path}/{file_path}',
                                                                                                                      'images'))]
    masks = [os.path.join(os.path.join(f'{data_path}/{file_path}', 'mask', f)) for f in os.listdir(os.path.join(f'{data_path}/{file_path}',
                                                                                                                   'mask'))]
    return images, masks

def load_data(path):
    train_valid_path = os.path.join('train')
    valid_path = os.path.join('validation')
    test_path = os.path.join('test')

    train_x, train_y = process_data(path, train_valid_path)
    valid_x, valid_y = process_data(path, valid_path)
    test_x, test_y = process_data(path, test_path)

    #train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)
    # train_y, valid_y = train_test_split(train_y, test_size=0.2, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

def read_image(x):
    #print(x)
    x = cv2.imread(x, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (W, H), interpolation=cv2.INTER_NEAREST)
    x = x.astype(np.float32)
    x = x / 255.0
    return x

def read_mask(x):
    #print(x)
    x = cv2.imread(x, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (W, H), interpolation=cv2.INTER_NEAREST)
    #x = x - 1
    x = x.astype(np.int32)
    return x


def tf_dataset(x, y, batch=8, width = 512, height = 512):
    global H
    global W
    W = width
    H = height
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.map(preprocess)
    dataset = dataset.map(lambda x, y: (tf.image.random_flip_left_right(x), y))
    #dataset = dataset.map(lambda x, y: (tf.image.random_flip_up_down(x), y))
    dataset = dataset.map(lambda x, y: (tf.image.random_crop(x, size=(width,height,3)), y))
    #dataset = dataset.map(lambda x, y: (tf.image.random_contrast(x, lower=0.5, upper=1.0), y))
    dataset = dataset.batch(batch)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def preprocess(x, y):
    def f(x, y):
        x = x.decode()
        y = y.decode()

        image = read_image(x)
        mask = read_mask(y)

        return image, mask

    image, mask = tf.numpy_function(f, [x, y], [tf.float32, tf.int32])
    mask = tf.one_hot(mask, num_classes, dtype=tf.float32)
    image.set_shape([H, W, 3])
    #image = image / 127.5 - 1
    mask.set_shape([H, W, num_classes])

    return image, mask

#
if __name__ == "__main__":
    path = "../dataset/"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path)
    print(f"Dataset: Train: {len(train_x)} - Valid: {len(valid_x)} - Test: {len(test_x)}")

    dataset = tf_dataset(train_x, train_y, batch=8)
    print(dataset)
    # for x, y in dataset:
    #     print(x.shape, y.shape) ## (8, 256, 256, 3), (8, 256, 256, 3)