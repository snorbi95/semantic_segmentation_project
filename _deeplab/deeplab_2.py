import matplotlib.pyplot as plt
import os
import random
import tensorflow as tf
import cv2
import numpy as np
import keras.backend as K


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


def dice_coef_loss(y_true, y_pred):
    from _losses.metrics import dice_coef
    return 1 - dice_coef(y_true, y_pred)


def dice_coef(y_true, y_pred, smooth=0.001):
    import numpy as np
    y_true_f = K.flatten(y_true[...,1:])
    y_pred_f = K.flatten(y_pred[...,1:])
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def jaccard_loss(y_true, y_pred):
    return (1 - jaccard_score_1(y_true, y_pred)) + (1 - jaccard_score_2(y_true, y_pred)) + (1 - jaccard_score_3(y_true, y_pred))# + categorical_focal_loss(alpha=[0.25,.25,.25,.25])(y_true, y_pred)


def jaccard_score(y_true, y_pred, smooth=0.001):
    import numpy as np
    y_true_f = K.flatten(y_true[...,1:])
    y_pred_f = K.flatten(y_pred[...,1:])
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f + y_pred_f) - intersection
    return (intersection) / (union + smooth)

def jaccard_score_1(y_true, y_pred, smooth=0.001):
    import numpy as np
    y_true_f = K.flatten(y_true[...,1])
    y_pred_f = K.flatten(y_pred[...,1])
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f + y_pred_f) - intersection
    return (intersection) / (union + smooth)

def jaccard_score_2(y_true, y_pred, smooth=0.001):
    import numpy as np
    y_true_f = K.flatten(y_true[...,2])
    y_pred_f = K.flatten(y_pred[...,2])
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f + y_pred_f) - intersection
    return (intersection) / (union + smooth)

def jaccard_score_3(y_true, y_pred, smooth=0.001):
    import numpy as np
    y_true_f = K.flatten(y_true[...,3])
    y_pred_f = K.flatten(y_pred[...,3])
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f + y_pred_f) - intersection
    return (intersection) / (union + smooth)


def jaccard_score_all(y_true, y_pred, smooth=0.001):
    import numpy as np
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f + y_pred_f) - intersection
    return (intersection) / (union + smooth)

img_size = (224, 224)
num_classes = 4
batch_size = 4


DATA_DIR = '../dataset/'

x_train_dir = os.path.join(DATA_DIR, 'train_crop/images')
y_train_dir = os.path.join(DATA_DIR, 'train_crop/mask')
train_len = len(os.listdir(x_train_dir))

x_valid_dir = os.path.join(DATA_DIR, 'validation_crop/images')
y_valid_dir = os.path.join(DATA_DIR, 'validation_crop/mask')
valid_len = len(os.listdir(x_valid_dir))

x_test_dir = os.path.join(DATA_DIR, 'test_crop/images')
y_test_dir = os.path.join(DATA_DIR, 'test_crop/mask')


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
    plt.savefig(f"results/4_classes/{fig_name}")
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

    CLASSES = ['background', 'arthery', 'ureter', 'nerve']

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

def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False,
):
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
    )(block_input)
    x = layers.BatchNormalization()(x)
    return tf.nn.relu(x)


def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output

def DeeplabV3Plus(image_size, num_classes):
    model_input = keras.Input(shape=(image_size, image_size, 3))
    resnet50 = keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_tensor=model_input
    )
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)

    input_a = layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same", activation='softmax')(x)
    return keras.Model(inputs=model_input, outputs=model_output)


# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()

BACKBONE = 'efficientnetb3'
BATCH_SIZE = batch_size
CLASSES = ['background', 'arthery', 'ureter', 'nerve']
LR = 0.00025
EPOCHS = 20

preprocess_input = sm.get_preprocessing(BACKBONE)

n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES))  # case for binary and multiclass segmentation
activation = 'sigmoid' if n_classes == 1 else 'softmax'

#create model
#model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)
#model = DeeplabV3Plus(img_size[0], num_classes)

optim = tf.keras.optimizers.Adam(LR)
from _losses import losses
# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
# set class weights for dice_loss (car: 1.; pedestrian: 2.; background: 0.5;)
#dice_loss = sm.losses.DiceLoss(class_weights=[0.5, 2, 2, 2])


total_loss = jaccard_loss

# actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
# total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss

metrics = [jaccard_score, jaccard_score_1, jaccard_score_2, jaccard_score_3, jaccard_score_all, dice_coef, losses.dice_coef_4cat]

#model.compile(optim, total_loss, metrics)
# compile keras model with defined optimozer, loss and metrics
model = tf.keras.models.load_model(f'models/deeplab_1_jaccard_loss_focal_224_size_full_3_epoch_extended_train_equal_cfc_weights.h5',
                                custom_objects={'jaccard_loss': jaccard_loss, 'jaccard_score': jaccard_score, 'jaccard_score_all': jaccard_score_all,
                                                'jaccard_score_1': jaccard_score_1, 'jaccard_score_2': jaccard_score_2,
                                                'jaccard_score_3': jaccard_score_3, 'dice_coef': dice_coef, 'dice_coef_4cat': losses.dice_coef_4cat})
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

train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True, length=train_len // batch_size )
valid_dataloader = Dataloder(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, length=valid_len // batch_size, validation=True)

#model desc
model_num = 1
loss = 'jaccard_loss_focal'
image_size = str(img_size[0])
image_mode = 'full'
add_info = 'extended_train_equal_cfc_weights'
model_name = f'deeplab_{model_num}_{loss}_{image_size}_size_{image_mode}_{EPOCHS}_epoch_{add_info}'

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

# Plot training & validation iou_score values
fig, ax = plt.subplots(2,2)

ax[0,0].plot(history.history["loss"])
ax[0,0].set_title("Training Loss")
ax[0,0].set_ylabel("loss")
ax[0,0].set_xlabel("epoch")

ax[0,1].plot(history.history["jaccard_score"])
ax[0,1].set_title("Training Accuracy")
ax[0,1].set_ylabel("accuracy")
ax[0,1].set_xlabel("epoch")

ax[1,0].plot(history.history["val_loss"])
ax[1,0].set_title("Validation Loss")
ax[1,0].set_ylabel("val_loss")
ax[1,0].set_xlabel("epoch")

ax[1,1].plot(history.history["val_jaccard_score"])
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

scores = model.evaluate(test_dataloader)

f = open(f'results/metrics/{model_name}.txt','w')
print("Loss: {:.5}".format(scores[0]), file=f)
for metric, value in zip(metrics, scores[1:]):
    print("mean {}: {:.5}".format(metric.__name__, value), file=f)
f.close()

n = 5
ids = len(test_dataloader)

for i in range(100):
    image, y = test_dataset[np.random.randint(0,ids)]
    image = np.expand_dims(image, axis=0)
    y = np.argmax(y, axis=-1)
    y = np.expand_dims(y, axis=-1)
    y = y * (255/num_classes)
    y = y.astype(np.int32)
    y = np.concatenate([y, y, y], axis=2)

    p = model.predict(image)
    p = np.argmax(p, axis=-1)
    p = np.expand_dims(p, axis=-1)
    p = p * (255/num_classes)
    p = p.astype(np.int32)
    p = np.concatenate([p, p, p], axis=3)

    visualize(
        fig_name=i,
        image=denormalize(image.squeeze()),
        gt_mask=y,
        pr_mask=p[0,:,:,:],
    )