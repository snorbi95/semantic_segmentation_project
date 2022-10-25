import os
import cv2
import numpy as np
from glob import glob
from scipy.io import loadmat
import matplotlib.pyplot as plt

import pandas as pd
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
import cv2
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def load_data(path):
    train_valid_path = os.path.join('train')
    test_path = os.path.join('test')

    train_x, train_y = process_data(path, train_valid_path)
    test_x, test_y = process_data(path, test_path)

    train_x, valid_x = train_test_split(train_x, test_size=0.2, random_state=42)
    train_y, valid_y = train_test_split(train_y, test_size=0.2, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

def process_data(data_path, file_path):

    images = [os.path.join(os.path.join(f'{data_path}/{file_path}', 'images', f)) for f in os.listdir(os.path.join(f'{data_path}/{file_path}',
                                                                                                                      'images'))]
    masks = [os.path.join(os.path.join(f'{data_path}/{file_path}', 'mask', f)) for f in os.listdir(os.path.join(f'{data_path}/{file_path}',
                                                                                                                   'mask'))]
    return images, masks

def read_image(image_path, mask=False):
    image = tf.io.read_file(image_path)
    if mask:
        image = tf.image.decode_jpeg(image, channels=1)
        image.set_shape([None, None, 1])
        image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE], method='nearest')
    else:
        image = tf.image.decode_png(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
        image = image / 255.0
    return image


def load_data_2(image_list, mask_list):
    image = read_image(image_list)
    mask = read_image(mask_list, mask=True)
    return image, mask


def data_generator(image_list, mask_list, batch_size, image_size):
    global BATCH_SIZE
    global IMAGE_SIZE
    BATCH_SIZE = batch_size
    IMAGE_SIZE = image_size
    dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
    dataset = dataset.map(load_data_2, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(lambda x, y: (tf.image.random_flip_left_right(x), y))
    dataset = dataset.map(lambda x, y: (tf.image.random_flip_up_down(x), y))
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.repeat()
    return dataset
