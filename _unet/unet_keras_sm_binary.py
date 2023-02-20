import math

import matplotlib.pyplot as plt
import os
import random
import tensorflow as tf
import cv2
import numpy as np
import keras.backend as K
from sklearn.utils.extmath import cartesian

current_weights = [0.05,  0.95]
def categorical_focal_loss(alpha, gamma=2.):
    """
    Softmax version of focal loss.
    When there is a skew between different categories/labels in your data set, you can try to apply this function as a
    loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy. Alpha is used to specify the weight of different
      categories/labels, the size of the array needs to be consistent with the number of classes.
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=[[.25, .25, .25]], gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    alpha = np.array(alpha, dtype=np.float32)

    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Compute mean loss in mini_batch
        return K.mean(K.sum(loss, axis=-1))

    return categorical_focal_loss_fixed

def create_weighted_binary_crossentropy(zero_weight, one_weight):

    def weighted_binary_crossentropy(y_true, y_pred):

        # Original binary crossentropy (see losses.py):
        # K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

        # Calculate the binary crossentropy
        b_ce = K.binary_crossentropy(y_true, y_pred)

        # Apply the weights
        weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
        weighted_b_ce = weight_vector * b_ce

        # Return the mean error
        return K.mean(weighted_b_ce)

    return weighted_binary_crossentropy

def cdist(A, B):
    # squared norms of each row in A and B
    na = tf.reduce_sum(tf.square(A), 1)
    nb = tf.reduce_sum(tf.square(B), 1)

    # na as a row and nb as a co"lumn vectors
    na = tf.reshape(na, [-1, 1])
    nb = tf.reshape(nb, [1, -1])

    # return pairwise euclidead difference matrix
    D = tf.sqrt(tf.maximum(na - 2 * tf.matmul(A, B, False, True) + nb, 0.0))
    return D


def weighted_hausdorff_distance(w, h, alpha):
    all_img_locations = tf.convert_to_tensor(cartesian([np.arange(w),
                                               np.arange(h)]), dtype=tf.float32)
    max_dist = math.sqrt(w ** 2 + h ** 2)

    def hausdorff_loss(y_true, y_pred):
        def loss(y_true, y_pred):
            eps = 1e-6
            y_true = K.reshape(y_true, [w, h])
            gt_points = K.cast(tf.where(y_true > 0.5), dtype=tf.float32)
            num_gt_points = tf.shape(gt_points)[0]
            y_pred = K.flatten(y_pred)
            p = y_pred
            p_replicated = tf.squeeze(K.repeat(tf.expand_dims(p, axis=-1),
                                                num_gt_points))
            d_matrix = cdist(all_img_locations, gt_points)
            num_est_pts = tf.reduce_sum(p)
            term_1 = (1 / (num_est_pts + eps)) * K.sum(p * K.min(d_matrix, 1))

            d_div_p = K.min((d_matrix + eps) / (p_replicated ** alpha + (eps / max_dist)), 0)
            d_div_p = K.clip(d_div_p, 0, max_dist)
            term_2 = K.mean(d_div_p, axis=0)

            return term_1 + term_2

        batched_losses = tf.map_fn(lambda x:
                                   loss(x[0], x[1]),
                                   (y_true, y_pred),
                                   dtype=tf.float32)
        return K.mean(tf.stack(batched_losses))

    return hausdorff_loss


def dice_coef_loss(y_true, y_pred):
    from _losses.metrics import dice_coef
    return 1 - dice_coef(y_true, y_pred)


def dice_coef(y_true, y_pred, smooth=0.001):
    import numpy as np
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def jaccard_loss(y_true, y_pred):
    #return ((1 - jaccard_score_1(y_true, y_pred)) + (1 - jaccard_score_2(y_true, y_pred)) + (1 - jaccard_score_3(y_true, y_pred)))# / 3 + (1 - jaccard_score(y_true, y_pred))# + categorical_focal_loss(alpha=[0.25,.25,.25,.25])(y_true, y_pred)
    return 1 - weighted_jaccard_score(y_true, y_pred, weights=current_weights)
    #return (1 - jaccard_score_true_class(y_true, y_pred, 1)) + (1 - jaccard_score_true_class(y_true, y_pred, 0))



def jaccard_score(y_true, y_pred, smooth=0.001):
    import numpy as np
    y_true_f = K.flatten(y_true[...,1:])
    y_pred_f = K.flatten(y_pred[...,1:])
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f + y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def weighted_jaccard_score(y_true, y_pred, weights):
    jaccard = 0
    for i, weight in enumerate(weights):
        jaccard += (weight * jaccard_score_class(y_true,y_pred,i))
    return jaccard / np.sum(current_weights)

def jaccard_score_class(y_true, y_pred, cl, smooth=0.001):
    import numpy as np
    y_true_f = K.flatten(y_true[...,cl])
    y_pred_f = K.flatten(y_pred[...,cl])
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f + y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def jaccard_score_true_class(y_true, y_pred, smooth=0.001):
    import numpy as np
    y_true_f = K.flatten(y_true[...,1])
    y_pred_f = K.flatten(y_pred[...,1])
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f + y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def jaccard_score_all(y_true, y_pred, smooth=0.001):
    import numpy as np
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f + y_pred_f) - intersection
    return (intersection) / (union + smooth)

img_size = (256, 256)
num_classes = 2
batch_size = 4
curr_class = 'arthery'


DATA_DIR = f'../{curr_class}_dataset/'

x_train_dir = os.path.join(DATA_DIR, 'train_crop_w_negative2/images')
y_train_dir = os.path.join(DATA_DIR, 'train_crop_w_negative2/mask')
train_len = len(os.listdir(x_train_dir))

x_valid_dir = os.path.join(DATA_DIR, 'test_crop_w_negative2/images')
y_valid_dir = os.path.join(DATA_DIR, 'test_crop_w_negative2/mask')
valid_len = len(os.listdir(x_valid_dir))

x_test_dir = os.path.join(DATA_DIR, 'test_crop_w_negative2/images')
y_test_dir = os.path.join(DATA_DIR, 'test_crop_w_negative2/mask')


def visualize(fig_name, **images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.savefig(f"results/{curr_class}_class/{fig_name}")
    plt.clf()


# helper function for data visualization
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x


# classes for data loading and preprocessing
class Dataset:
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    CLASSES = ['background', curr_class]

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i], cv2.IMREAD_COLOR)
        image = cv2.resize(image, (img_size[0], img_size[1]), interpolation=cv2.INTER_NEAREST)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #print(self.masks_fps[i])
        mask = cv2.imread(self.masks_fps[i], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (img_size[0], img_size[1]), interpolation=cv2.INTER_NEAREST)
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # fig, ax = plt.subplots(1,2)
        # ax[0].imshow(image)
        # ax[1].imshow(mask)
        # plt.show()

        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # # add background if mask is not binary
        # if mask.shape[-1] != 1:
        #     background = 1 - mask.sum(axis=-1, keepdims=True)
        #     mask = np.concatenate((mask, background), axis=-1)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)


class Dataloder(tf.keras.utils.Sequence):
    """Load data from dataset and form batches

    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """

    def __init__(self, dataset, batch_size=batch_size, shuffle=False, length = None, validation = None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))
        self.length = length
        self.validation = validation
        self.on_epoch_end()

    def __getitem__(self, i):

        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            if self.validation:
                data.append(self.dataset[j % valid_len])
            else:
                data.append(self.dataset[j % train_len])

        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]

        return batch

    def __len__(self):
        """Denotes the number of batches per epoch"""
        if self.length == None:
            return len(self.indexes) // self.batch_size
        else:
            return self.length

    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)
import albumentations as A

def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

# define heavy augmentations
def get_training_augmentation():
    train_transform = [

        A.HorizontalFlip(p=0.5),

        A.ShiftScaleRotate(scale_limit=0.6, rotate_limit=40, shift_limit=0.25, p=1, border_mode=0),

        A.PadIfNeeded(min_height=img_size[0], min_width=img_size[1], always_apply=True, border_mode=0),
        A.RandomCrop(height=img_size[0], width=img_size[1], always_apply=True),

        A.IAAAdditiveGaussianNoise(p=0.2),
        A.IAAPerspective(p=0.5),

        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightness(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.IAASharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.RandomContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
        A.Lambda(mask=round_clip_0_1)
    ]
    return A.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.PadIfNeeded(img_size[0], img_size[1])
    ]
    return A.Compose(test_transform)


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

from tensorflow.keras import layers
from tensorflow import keras
import segmentation_models as sm

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


# Free up RAM in case the model definition cells were run multiple times

weights = [[0.4,0.6]]
for weight in weights:
    keras.backend.clear_session()
    BACKBONE = 'efficientnetb3'
    current_weights = weight
    BATCH_SIZE = batch_size
    CLASSES = ['background', curr_class]
    LR = 0.001
    EPOCHS = 15

    preprocess_input = sm.get_preprocessing(BACKBONE)

    n_classes = 2# if len(CLASSES) == 1 else (len(CLASSES))  # case for binary and multiclass segmentation
    activation = 'sigmoid'# if n_classes == 1 else 'softmax'

    #create model
    model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)
    #model = get_model(img_size, num_classes)

    optim = tf.keras.optimizers.Adam(LR)
    from _losses import losses
    # Segmentation models losses can be combined together by '+' and scaled by integer or float factor
    # set class weights for dice_loss (car: 1.; pedestrian: 2.; background: 0.5;)
    #dice_loss = sm.losses.DiceLoss(class_weights=[0.5, 2, 2, 2])

    #weighted_hausdorff_distance(img_size[0], img_size[1], 0.001)#
    total_loss = jaccard_loss

    # actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
    # total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss

    metrics = [jaccard_score_true_class, jaccard_score_all, dice_coef]

    model.compile(optim, total_loss, metrics)
    # compile keras model with defined optimozer, loss and metrics

    model.summary()

    train_dataset = Dataset(
        x_train_dir,
        y_train_dir,
        classes=CLASSES,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocess_input),
    )

    # Dataset for validation images
    valid_dataset = Dataset(
        x_valid_dir,
        y_valid_dir,
        classes=CLASSES,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocess_input),
    )

    train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True, length=train_len // batch_size)
    valid_dataloader = Dataloder(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, length=valid_len // batch_size, validation=True)

    #model desc
    model_num = 1
    loss = 'weighted_jaccard_loss'
    image_size = str(img_size[0])
    image_mode = 'crop'
    add_info = f'binary_{curr_class}_w_true_weight_{current_weights[0]}_inc_lr'
    model_name = f'unet_{model_num}_{BACKBONE}_{loss}_{image_size}_size_{image_mode}_{EPOCHS}_epoch_{add_info}'
    # model = keras.models.load_model(f'models/unet_1_efficientnetb3_dice_loss_plus_focal_loss_224_size_crop_6_epoch_binary_arthery_w_true_weight_0.1_nadam.h5',
    #                                 custom_objects={'jaccard_loss': jaccard_loss,
    #                                                 'jaccard_score_true_class': jaccard_score_true_class,
    #                                                 'jaccard_score_all': jaccard_score_all,
    #                                                 'dice_coef': dice_coef})
    # define callbacks for learning rate scheduling and best checkpoints saving
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(f"models/{model_name}.h5", save_best_only=True, mode='min'),
        tf. keras.callbacks.ReduceLROnPlateau(),
        #tf.keras.callbacks.EarlyStopping(monitor="val_loss",min_delta=0,patience=10,verbose=0,mode="auto",baseline=None,restore_best_weights=False,)
    ]

    history = model.fit(
        train_dataloader,
        steps_per_epoch=len(train_dataloader),
        epochs=EPOCHS,
        callbacks=callbacks,
        validation_data=valid_dataloader,
        validation_steps=len(valid_dataloader),
    )

    #Plot training & validation iou_score values
    fig, ax = plt.subplots(2,2)

    ax[0,0].plot(history.history["loss"])
    ax[0,0].set_title("Training Loss")
    ax[0,0].set_ylabel("loss")
    ax[0,0].set_xlabel("epoch")

    ax[0,1].plot(history.history["jaccard_score_all"])
    ax[0,1].set_title("Training Accuracy")
    ax[0,1].set_ylabel("accuracy")
    ax[0,1].set_xlabel("epoch")

    ax[1,0].plot(history.history["val_loss"])
    ax[1,0].set_title("Validation Loss")
    ax[1,0].set_ylabel("val_loss")
    ax[1,0].set_xlabel("epoch")

    ax[1,1].plot(history.history["val_jaccard_score_all"])
    ax[1,1].set_title("Validation Accuracy")
    ax[1,1].set_ylabel("val_accuracy")
    ax[1,1].set_xlabel("epoch")
    plt.savefig(f'results/plots/model_plot_{model_name}.png', dpi = 300)

    test_dataset = Dataset(
        x_test_dir,
        y_test_dir,
        classes=CLASSES,
        #augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocess_input),
    )

    test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False)

    model = keras.models.load_model(f'models/{model_name}.h5',
                                     custom_objects={'jaccard_loss': jaccard_loss, 'jaccard_score_true_class': jaccard_score_true_class, 'jaccard_score_all': jaccard_score_all,
                                                      'dice_coef': dice_coef})
    scores = model.evaluate(test_dataloader)

    f = open(f'results/metrics/{model_name}.txt','w')
    print("Loss: {:.5}".format(scores[0]), file=f)
    for metric, value in zip(metrics, scores[1:]):
        print("mean {}: {:.5}".format(metric.__name__, value), file=f)
    f.close()

n = 5
ids = len(test_dataloader)
image, y = test_dataset[np.random.randint(0, ids)]
# plt.imshow(denormalize(image))
# plt.show()

# for i in range(100):
#     image, y = test_dataset[np.random.randint(0,ids)]
#     image = np.expand_dims(image, axis=0)
#     y = np.argmax(y, axis=-1)
#     y = np.expand_dims(y, axis=-1)
#     y = y * (255/num_classes)
#     y = y.astype(np.int32)
#     y = np.concatenate([y, y, y], axis=2)
#
#     p = model.predict(image)
#     p = np.argmax(p, axis=-1)
#     p = np.expand_dims(p, axis=-1)
#     p = p * (255/num_classes)
#     p = p.astype(np.int32)
#     p = np.concatenate([p, p, p], axis=3)
#
#     visualize(
#         fig_name=i,
#         image=denormalize(image.squeeze()),
#         gt_mask=y,
#         pr_mask=p[0,:,:,:],
#     )
# data_gen_args = dict(rescale=1./255,)
# image_datagen = ImageDataGenerator(**data_gen_args)
# mask_datagen = ImageDataGenerator(**data_gen_args)
# # image_datagen.fit(images)
# # mask_datagen.fit(masks)
# # Provide the same seed and keyword arguments to the fit and flow methods
# seed = 1
# image_generator = image_datagen.flow_from_directory(
#     '../dataset/train_gen/images',
#     batch_size=batch_size,
#     class_mode=None,
#     # color_mode='grayscale',
#     seed=seed)
# mask_generator = mask_datagen.flow_from_directory(
#     '../dataset/train_gen/mask',
#     batch_size=batch_size,
#     class_mode=None,
#     color_mode='grayscale',
#     seed=seed)
# # combine generators into one which yields image and masks
# train_generator = zip(image_generator, mask_generator)
#
# image_test_generator = image_datagen.flow_from_directory(
#     '../dataset/test_gen/images',
#     batch_size=batch_size,
#     class_mode=None,
#     # color_mode='grayscale',
#     seed=seed)
# mask_test_generator = mask_datagen.flow_from_directory(
#     '../dataset/test_gen/mask',
#     batch_size=batch_size,
#     class_mode=None,
#     color_mode='grayscale',
#     seed=seed)
# # combine generators into one which yields image and masks
# val_generator = zip(image_test_generator, mask_test_generator)
