import matplotlib.pyplot as plt
#import 'data.py' as data
#from keras.engine.data_adapter import DatasetAdapter
import os
import random
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from keras.preprocessing.image import ImageDataGenerator

#
input_dir = "../dataset/train_gen/images/"
target_dir = "../dataset/train_gen/mask/"

test_input_dir = '../dataset/validation_gen/images/'
test_target_dir = '../dataset/validation_gen/mask/'
img_size = (256, 256)
num_classes = 4
batch_size = 4
#(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data('../dataset')

import albumentations as A

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)


def AugmentGenerator(img_dir, mask_dir,seed=1, batch_size=batch_size):
    train_image_generator = train_frames_datagen.flow_from_directory(
        img_dir,
        batch_size=batch_size, seed=seed, target_size=img_size, interpolation='nearest')

    train_mask_generator = train_masks_datagen.flow_from_directory(
        mask_dir,
        batch_size=batch_size, seed=seed, target_size=img_size, color_mode='grayscale', interpolation='nearest')

    while True:
        X1i = train_image_generator.next()
        X2i = train_mask_generator.next()
        #
        # fig, ax = plt.subplots(1,2)
        # ax[0].imshow(X1i[0][0])
        # ax[1].imshow(X2i[0][0])
        # plt.show()
        # One hot encoding RGB images
        # image_encoded = [tf.image.per_image_standardization(X1i[0][x, :, :, :]) for x in range(X1i[0].shape[0])]
        # image_encoded = np.reshape(image_encoded, (X1i[0].shape[0], img_size[0], img_size[1], 3))

        mask_encoded = [tf.keras.utils.to_categorical(X2i[0][x, :, :, :], num_classes = num_classes) for x in range(X2i[0].shape[0])]
        mask_encoded = np.reshape(mask_encoded, (X2i[0].shape[0], img_size[0], img_size[1], num_classes))
        #mask_encoded = [rgb_to_onehot(X2i[0][x, :, :, :], id2code) for x in range(X2i[0].shape[0])]

        yield X1i[0], mask_encoded

def ValAugmentGenerator(seed=1, batch_size=batch_size):
    val_image_generator = val_frames_datagen.flow_from_directory(
        test_input_dir,
        batch_size=batch_size, seed=seed, target_size=img_size, shuffle=True)

    val_mask_generator = val_masks_datagen.flow_from_directory(
        test_target_dir,
        batch_size=batch_size, seed=seed, target_size=img_size, color_mode='grayscale', shuffle=True)

    while True:
        X1i = val_image_generator.next()
        X2i = val_mask_generator.next()

        # image_encoded = [tf.image.per_image_standardization(X1i[0][x, :, :, :]) for x in range(X1i[0].shape[0])]
        # image_encoded = np.reshape(image_encoded, (X1i[0].shape[0], img_size[0], img_size[1], 3))
        # fig, ax = plt.subplots(1,2)
        # ax[0].imshow(X1i[0][0])
        # ax[1].imshow(X2i[0][0])
        # plt.show()
        # One hot encoding RGB images
        mask_encoded = [tf.keras.utils.to_categorical(X2i[0][x, :, :, :], num_classes = num_classes) for x in range(X2i[0].shape[0])]
        mask_encoded = np.reshape(mask_encoded, (X2i[0].shape[0], img_size[0], img_size[1], num_classes))
        #mask_encoded = [rgb_to_onehot(X2i[0][x, :, :, :], id2code) for x in range(X2i[0].shape[0])]


        yield X1i[0], mask_encoded


from tensorflow import keras
import numpy as np
import segmentation_models as sm
from _losses import losses

from tensorflow.keras import layers


def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model


keras.backend.clear_session()

#model = get_model(img_size, num_classes)
BACKBONE = 'efficientnetb3'
BATCH_SIZE = batch_size
CLASSES = ['background', 'arthery', 'ureter', 'nerve']
LR = 0.0001
EPOCHS = 30

preprocess_input = sm.get_preprocessing(BACKBONE)

n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES))  # case for binary and multiclass segmentation
activation = 'sigmoid' if n_classes == 1 else 'softmax'

train_gen_args = dict(rescale = 1/ 255.0, zoom_range = 0.2, rotation_range = 30, width_shift_range = 0.2, height_shift_range = 0.2)
train_mask_gen_args = dict(zoom_range = 0.2, rotation_range = 30, width_shift_range = 0.2, height_shift_range = 0.2)
val_gen_args = dict(rescale = 1/ 255.0)
val_mask_gen_args = dict()


train_frames_datagen = ImageDataGenerator(**train_gen_args)
train_masks_datagen = ImageDataGenerator(**train_mask_gen_args)
val_frames_datagen = ImageDataGenerator(**val_gen_args)
val_masks_datagen = ImageDataGenerator(**val_mask_gen_args)


#create model
model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)
#model = get_model(img_size, num_classes)

optim = tf.keras.optimizers.Adam(LR)

dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

# compile keras model with defined optimozer, loss and metrics
model.compile(optim, total_loss, metrics)
epochs = 10
loss = 'dice_loss'

callbacks = [
    keras.callbacks.ModelCheckpoint(f"models/unet__loss_{loss}_{epochs}_epoch.h5", save_best_only=True)
]

#model = tf.keras.models.load_model("oxford_segmentation.h5")


train_steps = 512#len(train_x)//batch_size
valid_steps = 107//batch_size
#test_steps = len(test_x)//batch_size

#tf.keras.utils.plot_model(model, 'model.png', show_shapes = True)

#model.fit(train_generator, epochs=epochs, validation_data=val_generator, callbacks=callbacks, shuffle = True)
history = model.fit(AugmentGenerator(input_dir, target_dir), epochs=epochs, validation_data= AugmentGenerator(test_input_dir, test_target_dir), callbacks=callbacks, shuffle = True,steps_per_epoch=train_steps, validation_steps=valid_steps)
#model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), callbacks=callbacks, shuffle = True, class_weight = weight)

fig, ax = plt.subplots(2,2)

ax[0,0].plot(history.history["loss"])
ax[0,0].set_title("Training Loss")
ax[0,0].set_ylabel("loss")
ax[0,0].set_xlabel("epoch")

ax[0,1].plot(history.history["iou_score"])
ax[0,1].set_title("Training Accuracy")
ax[0,1].set_ylabel("accuracy")
ax[0,1].set_xlabel("epoch")

ax[1,0].plot(history.history["val_loss"])
ax[1,0].set_title("Validation Loss")
ax[1,0].set_ylabel("val_loss")
ax[1,0].set_xlabel("epoch")

ax[1,1].plot(history.history["val_iou_score"])
ax[1,1].set_title("Validation Accuracy")
ax[1,1].set_ylabel("val_accuracy")
ax[1,1].set_xlabel("epoch")
plt.savefig(f'results/model_plot.png', dpi = 300)
