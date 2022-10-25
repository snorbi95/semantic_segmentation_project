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

import tensorflow as tf


def vgg16(l2=0, dropout=0):
    '''Convolutionized VGG16 network.
    Args:
      l2 (float): L2 regularization strength
      dropout (float): Dropout rate
    Returns:
      (keras Model)
    '''
    ## Input
    input_layer = keras.Input(shape=(None, None, 3), name='input')
    ## Preprocessing
    x = keras.layers.Lambda(tf.keras.applications.vgg16.preprocess_input, name='preprocessing')(input_layer)
    ## Block 1
    x = keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu',
                            kernel_regularizer=keras.regularizers.L2(l2=l2), name='block1_conv1')(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=3,  strides=(1,1), padding='same', activation='relu',
                            kernel_regularizer=keras.regularizers.L2(l2=l2), name='block1_conv2')(x)
    x = keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid', name='block1_pool')(x)
    ## Block 2
    x = keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu',
                            kernel_regularizer=keras.regularizers.L2(l2=l2), name='block2_conv1')(x)
    x = keras.layers.Conv2D(filters=128, kernel_size=3,  strides=(1,1), padding='same', activation='relu',
                            kernel_regularizer=keras.regularizers.L2(l2=l2), name='block2_conv2')(x)
    x = keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid', name='block2_pool')(x)
    ## Block 3
    x = keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu',
                            kernel_regularizer=keras.regularizers.L2(l2=l2), name='block3_conv1')(x)
    x = keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu',
                            kernel_regularizer=keras.regularizers.L2(l2=l2), name='block3_conv2')(x)
    x = keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu',
                            kernel_regularizer=keras.regularizers.L2(l2=l2), name='block3_conv3')(x)
    x = keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid', name='block3_pool')(x)
    ## Block 4
    x = keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu',
                            kernel_regularizer=keras.regularizers.L2(l2=l2), name='block4_conv1')(x)
    x = keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu',
                            kernel_regularizer=keras.regularizers.L2(l2=l2), name='block4_conv2')(x)
    x = keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu',
                            kernel_regularizer=keras.regularizers.L2(l2=l2), name='block4_conv3')(x)
    x = keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid', name='block4_pool')(x)
    ## Block 5
    x = keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu',
                            kernel_regularizer=keras.regularizers.L2(l2=l2), name='block5_conv1')(x)
    x = keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu',
                            kernel_regularizer=keras.regularizers.L2(l2=l2), name='block5_conv2')(x)
    x = keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu',
                            kernel_regularizer=keras.regularizers.L2(l2=l2), name='block5_conv3')(x)
    x = keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid', name='block5_pool')(x)
    ## Convolutionized fully-connected layers
    x = keras.layers.Conv2D(filters=4096, kernel_size=(7,7), strides=(1,1), padding='same', activation='relu',
                            kernel_regularizer=keras.regularizers.L2(l2=l2), name='conv6')(x)
    x = keras.layers.Dropout(rate=dropout, name='drop6')(x)
    x = keras.layers.Conv2D(filters=4096, kernel_size=(1,1), strides=(1,1), padding='same', activation='relu',
                            kernel_regularizer=keras.regularizers.L2(l2=l2), name='conv7')(x)
    x = keras.layers.Dropout(rate=dropout, name='drop7')(x)
    ## Inference layer
    x = keras.layers.Conv2D(filters=1000, kernel_size=(1,1), strides=(1,1), padding='same', activation='softmax',
                            name='pred')(x)
    return keras.Model(input_layer, x)



def fcn32(vgg16, l2=0):
    '''32x upsampled FCN.
    Args:
      vgg16 (keras Model): VGG16 model to build upon
      l2 (float): L2 regularization strength
    Returns:
      (keras Model)
    '''
    x = keras.layers.Conv2D(filters=num_classes, kernel_size=(1,1), strides=(1,1), padding='same', activation='linear',
                            kernel_regularizer=keras.regularizers.L2(l2=l2),
                            name='score7')(vgg16.get_layer('drop7').output)
    x = keras.layers.Conv2DTranspose(filters=num_classes, kernel_size=(64,64), strides=(32,32),
                                     padding='same', use_bias=False, activation='softmax',
                                     kernel_initializer=BilinearInitializer(),
                                     kernel_regularizer=keras.regularizers.L2(l2=l2),
                                     name='fcn32')(x)
    return keras.Model(vgg16.input, x)



def fcn16(vgg16, fcn32, l2=0):
    '''16x upsampled FCN.
    Args:
      vgg16 (keras Model): VGG16 model to build upon
      fcn32 (keras Model): FCN32 model to build upon
      l2 (float): L2 regularization strength
    Returns:
      (keras Model)
    '''
    x = keras.layers.Conv2DTranspose(filters=num_classes, kernel_size=(4,4), strides=(2,2),
                                     padding='same', use_bias=False, activation='linear',
                                     kernel_initializer=BilinearInitializer(),
                                     kernel_regularizer=keras.regularizers.L2(l2=l2),
                                     name='score7_upsample')(fcn32.get_layer('score7').output)
    y = keras.layers.Conv2D(filters=num_classes, kernel_size=(1,1), strides=(1,1), padding='same', activation='linear',
                            kernel_initializer=keras.initializers.Zeros(),
                            kernel_regularizer=keras.regularizers.L2(l2=l2),
                            name='score4')(vgg16.get_layer('block4_pool').output)
    x = keras.layers.Add(name='skip4')([x, y])
    x = keras.layers.Conv2DTranspose(filters=num_classes, kernel_size=(32,32), strides=(16, 16),
                                     padding='same', use_bias=False, activation='softmax',
                                     kernel_initializer=BilinearInitializer(),
                                     kernel_regularizer=keras.regularizers.L2(l2=l2),
                                     name='fcn16')(x)
    return keras.Model(fcn32.input, x)



def fcn8(vgg16, fcn16, l2=0):
    '''8x upsampled FCN.
    Args:
      vgg16 (keras Model): VGG16 model to build upon
      fcn16 (keras Model): FCN16 model to build upon
      l2 (float): L2 regularization strength
    Returns:
      (keras Model)
    '''
    x = keras.layers.Conv2DTranspose(filters=num_classes, kernel_size=(4,4), strides=(2,2),
                                     padding='same', use_bias=False, activation='linear',
                                     kernel_initializer=BilinearInitializer(),
                                     kernel_regularizer=keras.regularizers.L2(l2=l2),
                                     name='skip4_upsample')(fcn16.get_layer('skip4').output)
    y = keras.layers.Conv2D(filters=num_classes, kernel_size=(1,1), strides=(1,1), padding='same', activation='linear',
                            kernel_initializer=keras.initializers.Zeros(),
                            kernel_regularizer=keras.regularizers.L2(l2=l2),
                            name='score3')(vgg16.get_layer('block3_pool').output)
    x = keras.layers.Add(name='skip3')([x, y])
    x = keras.layers.Conv2DTranspose(filters=num_classes, kernel_size=(16,16), strides=(8,8),
                                     padding='same', use_bias=False, activation='softmax',
                                     kernel_initializer=BilinearInitializer(),
                                     kernel_regularizer=keras.regularizers.L2(l2=l2),
                                     name='fcn8')(x)
    return keras.Model(fcn16.input, x)

class BilinearInitializer(keras.initializers.Initializer):
    '''Initializer for Conv2DTranspose to perform bilinear interpolation on each channel.'''
    def __call__(self, shape, dtype=None, **kwargs):
        kernel_size, _, filters, _ = shape
        arr = np.zeros((kernel_size, kernel_size, filters, filters))
        ## make filter that performs bilinear interpolation through Conv2DTranspose
        upscale_factor = (kernel_size+1)//2
        if kernel_size % 2 == 1:
            center = upscale_factor - 1
        else:
            center = upscale_factor - 0.5
        og = np.ogrid[:kernel_size, :kernel_size]
        kernel = (1-np.abs(og[0]-center)/upscale_factor) * \
                 (1-np.abs(og[1]-center)/upscale_factor) # kernel shape is (kernel_size, kernel_size)
        for i in range(filters):
            arr[..., i, i] = kernel
        return tf.convert_to_tensor(arr, dtype=dtype)


keras.backend.clear_session()

base_model = vgg16(l2=1e-6, dropout=0.2)
#vgg16 = keras.applications.vgg16.VGG16(weights='imagenet')
# weight_list = vgg16.get_weights()
# weight_list[26] = weight_list[26].reshape(7, 7, 512, 4096)
# weight_list[28] = weight_list[28].reshape(1, 1, 4096, 4096)
# weight_list[30] = weight_list[30].reshape(1, 1, 4096, 1000)
# base_model.set_weights(weight_list)
# del weight_list

fcn32 = fcn32(base_model, l2=1e-6)
fcn32.get_layer('fcn32').trainable=False

fcn16 = fcn16(base_model, fcn32, l2=1e-6)
fcn16.get_layer('score7_upsample').trainable=False
fcn16.get_layer('fcn16').trainable=False

fcn8 = fcn8(base_model, fcn16, l2=1e-6)
fcn8.get_layer('skip4_upsample').trainable=False
fcn8.get_layer('fcn8').trainable=False

model = fcn32
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