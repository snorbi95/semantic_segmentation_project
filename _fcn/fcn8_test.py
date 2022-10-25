import keras
from tensorflow.keras.layers import Conv2D, Dropout, Conv2DTranspose, Add, Softmax
from keras.layers.core import Activation
from keras.layers.convolutional import Cropping2D
from keras.models import Model
from keras.utils import plot_model

from keras.layers import Conv2D, Dropout, Input, MaxPooling2D
import keras

from keras import backend as K
import numpy as np

import tensorflow
from keras.layers import Flatten, Dense
from pylab import *
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from _unet import data

#from tensorflow.keras.engine import Layer
from tensorflow.keras.applications.vgg16 import *
from tensorflow.keras.models import *
#from tensorflow.keras.applications.imagenet_utils import _obtain_input_shape
import tensorflow.keras.backend as K
from tensorflow.keras.layers import MaxPooling2D, Cropping2D, Conv2D
from tensorflow.keras.layers import Input, Add, Dropout
from tensorflow import keras

# from tensorflow.compat.v1.layers import conv2d_transpose

input_dir = "../dataset/train/images/"
target_dir = "../dataset/train/mask/"

test_input_dir = '../dataset/test/images'
test_target_dir = '../dataset/test/mask'
img_size = (32, 32)
num_classes = 4
batch_size = 1


(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = data.load_data('../dataset')

train_dataset = data.tf_dataset(train_x, train_y, batch= batch_size, width = img_size[0], height = img_size[1])
val_dataset = data.tf_dataset(valid_x, valid_y, batch= batch_size, width = img_size[0], height = img_size[1])
test_dataset = data.tf_dataset(test_x, test_y, batch= batch_size, width = img_size[0], height = img_size[1])


def bilinear(shape, dtype=None):
    filter_size = shape[0]
    num_channels = shape[2]

    bilinear_kernel = np.zeros([filter_size, filter_size], dtype=np.float32)
    scale_factor = (filter_size + 1) // 2
    if filter_size % 2 == 1:
        center = scale_factor - 1
    else:
        center = scale_factor - 0.5
    for x in range(filter_size):
        for y in range(filter_size):
            bilinear_kernel[x, y] = (1 - abs(x - center) / scale_factor) * \
                                    (1 - abs(y - center) / scale_factor)
    weights = np.zeros((filter_size, filter_size, num_channels, num_channels))
    for i in range(num_channels):
        weights[:, :, i, i] = bilinear_kernel

    return weights

def vgg_encoder( shape ):

    img_input = Input(shape)

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1' )(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2' )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool' )(x)
    f1 = x
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1' )(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2' )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool' )(x)
    f2 = x

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1' )(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2' )(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3' )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool' )(x)
    f3 = x

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1' )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2' )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3' )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool' )(x)
    f4 = x

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1' )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2' )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3' )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool' )(x)
    f5 = x

    return img_input , [f1 , f2 , f3 , f4 , f5 ]


class fcn8(object):

    def __init__(self, n_classes, shape):
        self.n_classes = n_classes
        self.shape = shape

    def get_model(self):
        n_classes = self.n_classes
        shape = self.shape

        img_input, [f1, f2, f3, f4, f5] = vgg_encoder(shape)

        o = f5
        o = Conv2D(4096, (7, 7), activation='relu', padding='same')(o)
        o = Dropout(0.5)(o)
        o = Conv2D(4096, (1, 1), activation='relu', padding='same')(o)
        o = Dropout(0.5)(o)

        o = Conv2D(n_classes, (1, 1), activation='relu')(o)
        o = Conv2DTranspose(n_classes, kernel_size=(4, 4), strides=(2, 2), use_bias=False, kernel_initializer=bilinear)(
            o)
        o = Cropping2D(((1, 1), (1, 1)))(o)

        o2 = f4
        o2 = Conv2D(n_classes, (1, 1), activation='relu')(o2)

        o = Add()([o, o2])
        o = Conv2DTranspose(n_classes, kernel_size=(4, 4), strides=(2, 2), use_bias=False)(o)
        o = Cropping2D(((1, 1), (1, 1)))(o)

        o2 = f3
        o2 = Conv2D(n_classes, (1, 1), activation='relu')(o2)

        o = Add()([o2, o])
        o = Conv2DTranspose(n_classes, kernel_size=(16, 16), strides=(8, 8), use_bias=False)(o)
        o = Cropping2D(((4, 4), (4, 4)))(o)
        o = Softmax(axis=3)(o)

        model = Model(img_input, o)
        model.model_name = "fcn_8"
        return model


x = fcn8(shape=(img_size[0], img_size[1], 3), n_classes=num_classes)
model = x.get_model()
model.summary()

epochs = 30
train_steps = len(train_x)//batch_size
valid_steps = len(valid_x)//batch_size

model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])


model.summary()

result = model.fit(train_dataset,
                    verbose=1,
                    epochs=10,
                    validation_data=val_dataset,
                    shuffle=True,steps_per_epoch=train_steps, validation_steps=valid_steps)