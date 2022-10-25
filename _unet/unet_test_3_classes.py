import csv
import math
import operator

import matplotlib.pyplot as plt
import skimage.color
from sklearn.utils.extmath import cartesian

from _unet import data
import keras
from keras.engine.data_adapter import DatasetAdapter
import os
import random
import segmentation_models as sm
import tensorflow as tf
import numpy as np
import cv2
import keras.backend as K

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
            masks_dir = None,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        if masks_dir:
            self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        else:
            self.masks_fps = None

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i], cv2.IMREAD_COLOR)
        image = cv2.resize(image, (model_img_size[0], model_img_size[1]), interpolation=cv2.INTER_NEAREST)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #print(self.masks_fps[i])
        if self.masks_fps:
            mask = cv2.imread(self.masks_fps[i], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (model_img_size[0], model_img_size[1]), interpolation=cv2.INTER_NEAREST)
            masks = [(mask == v) for v in self.class_values]
            mask = np.stack(masks, axis=-1).astype('float')
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # fig, ax = plt.subplots(1,2)
        # ax[0].imshow(image)
        # ax[1].imshow(mask)
        # plt.show()

        # extract certain classes from mask (e.g. cars)

        # # add background if mask is not binary
        # if mask.shape[-1] != 1:
        #     background = 1 - mask.sum(axis=-1, keepdims=True)
        #     mask = np.concatenate((mask, background), axis=-1)

        # apply augmentations
        if self.augmentation:
            if self.masks_fps:
                sample = self.augmentation(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']
            else:
                sample = self.augmentation(image=image)
                image = sample['image']

        # apply preprocessing
        if self.preprocessing:
            if self.masks_fps:
                sample = self.preprocessing(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']
            else:
                sample = self.preprocessing(image=image)
                image = sample['image']

        if self.masks_fps:
            return image, mask
        else:
            return image

    def __len__(self):
        return len(self.ids)


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



def dice_coef(y_true, y_pred, smooth=0.001):
    import numpy as np
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def jaccard_loss(y_true, y_pred):
    #return ((1 - jaccard_score_1(y_true, y_pred)) + (1 - jaccard_score_2(y_true, y_pred)) + (1 - jaccard_score_3(y_true, y_pred)))# / 3 + (1 - jaccard_score(y_true, y_pred))# + categorical_focal_loss(alpha=[0.25,.25,.25,.25])(y_true, y_pred)
    return 1 - weighted_jaccard_score(y_true, y_pred, weights=[0.001,  0.999])


def weighted_jaccard_score(y_true, y_pred, weights):
    jaccard = 0
    for i, weight in enumerate(weights):
        jaccard += (weight * jaccard_score_class(y_true,y_pred,i))
    return jaccard


def jaccard_score_0(y_true, y_pred, smooth=0.001):
    import numpy as np
    y_true_f = K.flatten(y_true[...,0])
    y_pred_f = K.flatten(y_pred[...,0])
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f + y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def jaccard_score_1(y_true, y_pred, smooth=0.001):
    import numpy as np
    y_true_f = K.flatten(y_true[...,1])
    y_pred_f = K.flatten(y_pred[...,1])
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f + y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def jaccard_score_2(y_true, y_pred, smooth=0.001):
    import numpy as np
    y_true_f = K.flatten(y_true[...,2])
    y_pred_f = K.flatten(y_pred[...,2])
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f + y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def jaccard_score_3(y_true, y_pred, smooth=0.001):
    import numpy as np
    y_true_f = K.flatten(y_true[...,3])
    y_pred_f = K.flatten(y_pred[...,3])
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f + y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

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
    return (intersection + smooth) / (union + smooth)


def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x


img_size = (512, 512)
model_img_size = (256, 256)
num_classes = 4

nerve_model_dir = f'models/binary_nerve/default_dataset_distribution'
artery_model_dir = f'models/binary_arthery/default_dataset_distribution'
ureter_model_dir = f'models/binary_ureter/default_dataset_distribution'

nerve_models = [names for names in os.listdir(nerve_model_dir) if not os.path.isdir(names)]
artery_models = [names for names in os.listdir(artery_model_dir) if not os.path.isdir(names)]
ureter_models = [names for names in os.listdir(ureter_model_dir) if not os.path.isdir(names)]

multiclass_model = keras.models.load_model(f'models/multiclass/default_dataset_distribution/unet_1_weighted_jaccard_256_size_crop_20_epoch_multiclass_inc_lr.h5',
                                custom_objects={'jaccard_loss': jaccard_loss, 'weighted_jaccard_score': weighted_jaccard_score,
                                                'jaccard_score_0': jaccard_score_0, 'jaccard_score_1': jaccard_score_1, 'jaccard_score_2': jaccard_score_2,
                                                'jaccard_score_3':jaccard_score_3,'jaccard_score_all': jaccard_score_all,
                                                'dice_coef': dice_coef})

from skimage import io

BACKBONE = 'efficientnetb3'
preprocess_input = sm.get_preprocessing(BACKBONE)
CLASSES = ['background', 'arthery', 'ureter', 'nerve']

test_dir = f'../dataset/test_dataset_crop/images'
test_mask_dir = f'../dataset/test_dataset_crop/mask'
test_frame_names = os.listdir(test_dir)
test_dataset = Dataset(
    test_dir,
    masks_dir=test_mask_dir,
    classes=CLASSES,
    #augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocess_input),
)
# plt.imshow(test_dataset[0])
# plt.show()

indices = np.zeros((model_img_size[0],model_img_size[1],3))
values = np.zeros((model_img_size[0],model_img_size[1],3))


def region_prediction(preds):
    from skimage.measure import regionprops, label
    # plt.imshow(values[:,:,1])
    # plt.show()
    preds = np.copy(preds)
    prediction_map = np.zeros_like(preds)
    prediction_map[preds != 0] = 1
    label_image = label(prediction_map)
    regions = regionprops(label_image)
    for reg in regions:
        pred_region_image = preds[reg.bbox[0]:reg.bbox[2],reg.bbox[1]:reg.bbox[3]]
        values = sorted(np.unique(pred_region_image))
        print(values)
        values_dict = {}
        for value in values:
            if value != 0:
                values_dict[value] = pred_region_image[pred_region_image == value].size
        print(values_dict)
        max_value = max(values_dict.items(), key=operator.itemgetter(1))[0]
        pred_region_image[pred_region_image != 0] = max_value
        preds[reg.bbox[0]:reg.bbox[2],reg.bbox[1]:reg.bbox[3]] = pred_region_image
    return preds
    # plt.imshow(prediction)
    # plt.show()


def region_prediction_mean_confidence(preds, indices_mtx, values_mtx):
    preds = np.copy(preds)
    from skimage.measure import regionprops, label
    # plt.imshow(values[:,:,1])
    # plt.show()
    prediction_map = np.zeros_like(preds)
    prediction_map[preds != 0] = 1
    label_image = label(prediction_map)
    regions = regionprops(label_image)
    for reg in regions:
        pred_region_image = preds[reg.bbox[0]:reg.bbox[2],reg.bbox[1]:reg.bbox[3]]
        values = sorted(np.unique(pred_region_image))
        print(values)
        values_dict = {}
        for value in values:
            if value != 0:
                region_confidence_values = values_mtx[reg.bbox[0]: reg.bbox[2],reg.bbox[1]:reg.bbox[3], int(value) - 1]
                region_indice_values = indices_mtx[reg.bbox[0]: reg.bbox[2],reg.bbox[1]:reg.bbox[3], int(value) - 1]
                mean_confidence = region_confidence_values[region_indice_values != 0].mean()
                if mean_confidence > 0.9:
                    print('Region dropped...')
                    values_dict[value] = pred_region_image[pred_region_image == value].size
        print(values_dict)
        if len(values_dict.items()) > 0:
            max_value = max(values_dict.items(), key=operator.itemgetter(1))[0]
            pred_region_image[pred_region_image != 0] = max_value
            preds[reg.bbox[0]:reg.bbox[2],reg.bbox[1]:reg.bbox[3]] = pred_region_image
    return preds
    # plt.imshow(prediction)
    # plt.show()
extracted_models = [(0,0,0),(0,0,1), (0,0,2), (0,0,3), (0,0,4), (1,0,0), (1,0,1), (1,0,2), (1,0,3), (1,0,4),(2,0,0), (2,0,1), (2,0,2), (2,0,3), (2,0,4), (3, 0, 0),
                     (3, 0, 1), (3, 0, 2), (3, 0, 3), (3, 0, 4), (4, 0, 0), (4, 0, 1), (4, 0, 2), (4, 0, 3), (4, 0, 4), (0, 1, 0), (0, 1, 1),
                     (0,1,2), (0,1,3), (0, 1, 4), (1,1,0), (1,1,1), (1,1,2), (1,1,3), (1,1,4), (2,1,0), (2,1,1), (2,1,2), (2,1,3), (2,1,4),(3,1,0),(3,1,1), (3,1,2),
                     (3,1,3), (3,1,4), (4,1,0), (4,1,1),(4,1,2),(4,1,3),(4,1,4),(0,2,0),(0,2,1), (0,2,2), (0,2,3), (0,2,4), (1,2,0),(1,2,1), (1,2,2), (1,2,3),(1,2,4),
                     (2,2,0),(2,2,1),(2,2,2), (2,2,3), (2,2,4), (3,2,0),(3,2,1),(3,2,2),(3,2,3), (3,2,4), (4,2,0), (4,2,1),(4,2,2), (4,2,3), (4,2,4), (0,3,0), (0,3,1), (0,3,2), (0,3,3),
                     (0,3,4), (1,3,0), (1,3,1), (1,3,2), (1,3,3), (1,3,4), (2,3,0),(2,3,1),(2,3,2),(2,3,3),(2,3,4),(3,3,0),(3,3,1),(3,3,2),(3,3,3),(3,3,4),(4,3,0),(4,3,1),(4,3,2),(4,3,3),
                     (4,3,4),(0,4,0),(0,4,1),(0,4,2),(0,4,3),(0,4,4),(1,4,0),(1,4,1),(1,4,2),(1,4,3),(1,4,4),(2,4,0),(2,4,1)]#,(2,4,2)]#+,(2,4,3),(2,4,4),(3,4,0),(3,4,1),(3,4,2), (3,4,3),(3,4,4)
#                    (4,4,0),(4,4,1),(4,4,2),(4,4,3),(4,4,4)]

from numba import cuda
device = cuda.get_current_device()
for ureter_num, ureter_model_name in enumerate(ureter_models):
    for artery_num, artery_model_name in enumerate(artery_models):
        for nerve_num, nerve_model_name in enumerate(nerve_models):
            keras.backend.clear_session()
            if (artery_num, ureter_num, nerve_num) in extracted_models:
                continue
            indices = np.zeros((model_img_size[0], model_img_size[1], 3))
            values = np.zeros((model_img_size[0], model_img_size[1], 3))
            print(f'Predict using #{artery_num} artery - #{ureter_num}ureter - #{nerve_num} nerve')
            nerve_model = keras.models.load_model(f'{nerve_model_dir}/{nerve_model_name}',
                                            custom_objects={'jaccard_loss': jaccard_loss, 'jaccard_score_true_class': jaccard_score_true_class, 'jaccard_score_all': jaccard_score_all,
                                                            'dice_coef': dice_coef})

            ureter_model = keras.models.load_model(f'{ureter_model_dir}/{ureter_model_name}',
                                            custom_objects={'jaccard_loss': jaccard_loss, 'jaccard_score_true_class': jaccard_score_true_class, 'jaccard_score_all': jaccard_score_all,
                                                            'dice_coef': dice_coef})

            arthery_model = keras.models.load_model(f'{artery_model_dir}/{artery_model_name}',
                                            custom_objects={'jaccard_loss': jaccard_loss, 'jaccard_score_true_class': jaccard_score_true_class, 'jaccard_score_all': jaccard_score_all,
                                                            'dice_coef': dice_coef})

            jaccard_score_pixelwide = 0
            jaccard_score_region_based = 0
            jaccard_score_multiclass = 0

            dice_score_pixelwide = 0
            dice_score_region_based = 0
            dice_score_multiclass = 0

            jaccard_score_arthery_pixelwide = 0
            jaccard_score_arthery_region_based = 0
            jaccard_score_arthery_multiclass = 0

            jaccard_score_ureter_pixelwide = 0
            jaccard_score_ureter_region_based = 0
            jaccard_score_ureter_multiclass = 0

            jaccard_score_nerve_pixelwide = 0
            jaccard_score_nerve_region_based = 0
            jaccard_score_nerve_multiclass = 0

            for n in range(len(test_dataset)):
                test_frame_name, y = test_dataset[n]
                print(f'Predicting image #{n}...')
                # random_x = np.random.randint(0, test_frame.shape[0] - img_size[0])
                # random_y = np.random.randint(0, test_frame.shape[1] - img_size[1])
                # test_frame = test_frame[random_x:random_x + img_size[0],random_y:random_y + img_size[1]]
                test_frame = cv2.resize(test_frame_name, model_img_size, interpolation=cv2.INTER_NEAREST)
                # plt.imshow(test_frame)
                # plt.show()

                p1 = arthery_model.predict(np.expand_dims(test_frame, axis=0))
                #p1 = np.argmax(p1, axis=-1)
                indices_p1 = np.argmax(p1, axis=-1)
                values_p1 = np.max(p1, axis=-1)
                indices_p1 = indices_p1[0, :, :]
                values_p1 = values_p1[0, :, :]
                indices[:,:,0] = indices_p1
                values[:,:,0] = values_p1
                # p1 = np.expand_dims(p1, axis=-1)
                # p1 = p1 * (255 / num_classes)
                # p1 = p1.astype(np.int32)
                # p1 = np.concatenate([p1, p1, p1], axis=3)

                p2 = ureter_model.predict(np.expand_dims(test_frame, axis=0))
                indices_p2 = np.argmax(p2, axis=-1)
                indices_p2[indices_p2 != 0] = 2
                values_p2 = np.max(p2, axis=-1)
                indices_p2 = indices_p2[0, :, :]
                values_p2 = values_p2[0, :, :]
                indices[:,:,1] = indices_p2
                values[:,:,1] = values_p2
                # p2 = np.argmax(p2, axis=-1)
                # p2 = np.expand_dims(p2, axis=-1)
                # p2 = p2 * (255 / num_classes)
                # p2 = p2.astype(np.int32)
                # p2 = np.concatenate([p2, p2, p2], axis=3)

                p3 = nerve_model.predict(np.expand_dims(test_frame, axis=0))
                indices_p3 = np.argmax(p3, axis=-1)
                indices_p3[indices_p3 != 0] = 3
                values_p3 = np.max(p3, axis=-1)
                indices_p3 = indices_p3[0, :, :]
                values_p3 = values_p3[0, :, :]
                indices[:,:,2] = indices_p3
                values[:,:,2] = values_p3

                #nerve
                print(np.max(np.argmax(y, axis= - 1)))
                if np.max(np.argmax(y, axis= - 1)) == 3:
                    indices[:, :, 1] = np.zeros_like(indices_p2)
                    values[:, :, 1] = np.zeros_like(values_p2)

                    indices[:,:,0] = np.zeros_like(indices_p1)
                    values[:,:,0] = np.zeros_like(values_p1)
                # p3 = np.argmax(p3, axis=-1)
                # p3 = np.expand_dims(p3, axis=-1)
                # p3 = p3 * (255 / num_classes)
                # p3 = p3.astype(np.int32)
                # p3 = np.concatenate([p3, p3, p3], axis=3)

                # p_multiclass = multiclass_model.predict(np.expand_dims(test_frame, axis=0))
                # p_multiclass = np.argmax(p_multiclass, axis=-1)
                # p_multiclass = np.expand_dims(p_multiclass, axis=-1)
                # p_multiclass = p_multiclass * (255 / num_classes)
                # p_multiclass = p_multiclass.astype(np.int32)
                # p_multiclass = np.concatenate([p_multiclass, p_multiclass, p_multiclass], axis=3)

                prediction = np.zeros(model_img_size)

                for i in range(model_img_size[0]):
                    for j in range(model_img_size[0]):
                        if np.sum(indices[i, j]) == 0:
                            prediction[i, j] = 0
                        else:
                            max_value = 0
                            max_class = 0
                            for k in range(3):
                                if values[i, j, k] > max_value and indices[i, j, k] != 0:
                                    max_class = indices[i, j, k]
                                    max_value = values[i, j, k]
                            prediction[i, j] = max_class

                #print(jaccard_score_all(y, tf.keras.utils.to_categorical(prediction, num_classes= 4).astype('float64')))
                reg_pred_1 = region_prediction(np.copy(prediction))
                reg_pred_2 = region_prediction_mean_confidence(np.copy(prediction), indices, values)
                # fig, ax = plt.subplots(2,4)
                # ax[0,0].imshow(denormalize(test_frame))
                # ax[0,1].imshow(indices_p1* (255/ 4), vmax = 255.0)
                # ax[0,1].set_title('Arthery')
                # ax[1,0].imshow(indices_p2* (255/ 4), vmax = 255.0)
                # ax[1,0].set_title('Ureter')
                # ax[1,1].imshow(indices_p3* (255/ 4), vmax = 255.0)
                # ax[1,1].set_title('Nerve')
                # ax[0,2].imshow(prediction* (255/ 4), vmax = 255.0)
                # ax[0,2].set_title('Combined prediction')
                # ax[1,2].imshow(np.argmax(y, axis=-1) * (255/ 4), vmax = 255.0)
                # ax[1,2].set_title('Ground truth')
                # ax[0,3].imshow(reg_pred_1 * (255/ 4), vmax = 255.0)
                # ax[0,3].set_title('Region prediction')
                # # ax[1,3].imshow(p_multiclass[0,:], vmax = 255.0)
                # # ax[1,3].set_title('Multiclass prediction')
                # #plt.show()
                # # # ax[2,0].imshow(p_multiclass[0,:,:,:] * (255/ 4), vmax = 255.0)
                # # # ax[2,0].set_title('Multiclass prediction')
                #p_multiclass = np.max(p_multiclass, axis=-1) / (255 / 4)
                # plt.savefig(f'results/4_classes/combined_preds/{n}_all.png')
                # plt.clf()

                #print(weighted_hausdorff_distance(model_img_size[0], model_img_size[1],0.001)(y, tf.keras.utils.to_categorical(prediction, num_classes= 4).astype('float64')))

                jaccard_score_pixelwide += K.eval(jaccard_score_all(y, tf.keras.utils.to_categorical(prediction, num_classes= 4).astype('float64')))
                jaccard_score_region_based += K.eval(jaccard_score_all(y, tf.keras.utils.to_categorical(reg_pred_1, num_classes=4).astype('float64')))
                # jaccard_score_multiclass += K.eval(jaccard_score_all(y, tf.keras.utils.to_categorical(p_multiclass, num_classes=4).astype('float64')))

                dice_score_pixelwide += K.eval(dice_coef(y, tf.keras.utils.to_categorical(prediction, num_classes= 4).astype('float64')))
                dice_score_region_based += K.eval(dice_coef(y, tf.keras.utils.to_categorical(reg_pred_1, num_classes=4).astype('float64')))
                # dice_score_multiclass += K.eval(dice_coef(y, tf.keras.utils.to_categorical(p_multiclass, num_classes=4).astype('float64')))

                if np.max(np.argmax(y, axis= - 1)) == 3:
                    jaccard_score_arthery_pixelwide += 0#K.eval(jaccard_score_1(y, tf.keras.utils.to_categorical(prediction, num_classes= 4).astype('float64')))
                    jaccard_score_arthery_region_based += 0#K.eval(jaccard_score_1(y, tf.keras.utils.to_categorical(reg_pred_1, num_classes=4).astype('float64')))
                    jaccard_score_arthery_multiclass += 0#K.eval(jaccard_score_1(y, tf.keras.utils.to_categorical(p_multiclass, num_classes=4).astype('float64')))

                    jaccard_score_ureter_pixelwide += 0#K.eval(jaccard_score_2(y, tf.keras.utils.to_categorical(prediction, num_classes= 4).astype('float64')))
                    jaccard_score_ureter_region_based += 0#K.eval(jaccard_score_2(y, tf.keras.utils.to_categorical(reg_pred_1, num_classes=4).astype('float64')))
                    jaccard_score_ureter_multiclass += 0#K.eval(jaccard_score_2(y, tf.keras.utils.to_categorical(p_multiclass, num_classes=4).astype('float64')))
                else:
                    jaccard_score_arthery_pixelwide += K.eval(
                        jaccard_score_1(y, tf.keras.utils.to_categorical(prediction, num_classes=4).astype('float64')))
                    jaccard_score_arthery_region_based += K.eval(
                        jaccard_score_1(y, tf.keras.utils.to_categorical(reg_pred_1, num_classes=4).astype('float64')))
                    # jaccard_score_arthery_multiclass += K.eval(jaccard_score_1(y, tf.keras.utils.to_categorical(
                    #     p_multiclass, num_classes=4).astype('float64')))

                    jaccard_score_ureter_pixelwide += K.eval(
                        jaccard_score_2(y, tf.keras.utils.to_categorical(prediction, num_classes=4).astype('float64')))
                    jaccard_score_ureter_region_based += K.eval(
                        jaccard_score_2(y, tf.keras.utils.to_categorical(reg_pred_1, num_classes=4).astype('float64')))
                    # jaccard_score_ureter_multiclass += K.eval(jaccard_score_2(y, tf.keras.utils.to_categorical(
                    #     p_multiclass, num_classes=4).astype('float64')))

                jaccard_score_nerve_pixelwide += K.eval(jaccard_score_3(y, tf.keras.utils.to_categorical(prediction, num_classes= 4).astype('float64')))
                jaccard_score_nerve_region_based += K.eval(jaccard_score_3(y, tf.keras.utils.to_categorical(reg_pred_1, num_classes=4).astype('float64')))
                # jaccard_score_nerve_multiclass += K.eval(jaccard_score_3(y,tf.keras.utils.to_categorical(p_multiclass,num_classes=4).astype('float64')))
                del y
                del test_frame


            jaccard_score_pixelwide = jaccard_score_pixelwide / len(test_dataset)
            jaccard_score_region_based = jaccard_score_region_based / len(test_dataset)
            jaccard_score_multiclass = jaccard_score_multiclass / len(test_dataset)

            dice_score_pixelwide = dice_score_pixelwide / len(test_dataset)
            dice_score_region_based = dice_score_region_based / len(test_dataset)
            dice_score_multiclass = dice_score_multiclass / len(test_dataset)

            jaccard_score_arthery_pixelwide = jaccard_score_arthery_pixelwide / (len(test_dataset) - 228)
            jaccard_score_arthery_region_based = jaccard_score_arthery_region_based / (len(test_dataset) - 228)
            jaccard_score_arthery_multiclass = jaccard_score_arthery_multiclass / (len(test_dataset) - 228)

            jaccard_score_ureter_pixelwide = jaccard_score_ureter_pixelwide / (len(test_dataset) - 228)
            jaccard_score_ureter_region_based = jaccard_score_ureter_region_based / (len(test_dataset) - 228)
            jaccard_score_ureter_multiclass = jaccard_score_ureter_multiclass / (len(test_dataset) - 228)

            jaccard_score_nerve_pixelwide = jaccard_score_nerve_pixelwide / len(test_dataset)
            jaccard_score_nerve_region_based = jaccard_score_nerve_region_based / len(test_dataset)
            jaccard_score_nerve_multiclass = jaccard_score_nerve_multiclass / len(test_dataset)

            with open(f'results/ensemble_results_default.csv', 'a', newline='') as f:
                writer = csv.writer(f, delimiter = ',')
                writer.writerow([f'{artery_num}_{ureter_num}_{nerve_num}',jaccard_score_arthery_pixelwide, jaccard_score_arthery_region_based, jaccard_score_ureter_pixelwide, jaccard_score_ureter_region_based
                                 , jaccard_score_nerve_pixelwide, jaccard_score_nerve_region_based, jaccard_score_pixelwide, jaccard_score_region_based, dice_score_pixelwide, dice_score_region_based])
            print(f'Pixelwide jaccard score: {jaccard_score_pixelwide}')
            print(f'Region based jaccard score: {jaccard_score_region_based}')
            print(f'Multiclass jaccard score: {jaccard_score_multiclass}\n')

            print(f'Pixelwide dice score: {dice_score_pixelwide}')
            print(f'Region based dice score: {dice_score_region_based}')
            print(f'Multiclass dice score: {dice_score_multiclass}\n')

            print(f'Pixelwide jaccard score of arthery: {jaccard_score_arthery_pixelwide}')
            print(f'Region based jaccard score of arthery: {jaccard_score_arthery_region_based}')
            print(f'Multiclass jaccard score of arthery: {jaccard_score_arthery_multiclass}\n')

            print(f'Pixelwide jaccard score of ureter: {jaccard_score_ureter_pixelwide}')
            print(f'Region based jaccard score of ureter: {jaccard_score_ureter_region_based}')
            print(f'Multiclass jaccard score of ureter: {jaccard_score_ureter_multiclass}\n')

            print(f'Pixelwide jaccard score of nerve: {jaccard_score_nerve_pixelwide}')
            print(f'Region based jaccard score of nerve: {jaccard_score_nerve_region_based}')
            print(f'Multiclass jaccard score of nerve: {jaccard_score_nerve_multiclass}\n')

            del nerve_model
            del arthery_model
            del ureter_model
            del reg_pred_1
            del reg_pred_2
            del indices
            del values