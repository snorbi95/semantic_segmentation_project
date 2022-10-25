from keras.losses import binary_crossentropy
import keras.backend as K
import tensorflow as tf

epsilon = 1e-5
smooth = 1
num_classes = 4

def dsc(y_true, y_pred):
    smooth = 1.
    y_true_f = y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=num_classes)[...,1:])
    y_pred_f = K.flatten(y_pred[...,1:])
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def dice_coef_4cat(y_true, y_pred, smooth=1e-7):
    '''
    Dice coefficient for 10 categories. Ignores background pixel label 0
    Pass to model as metric during compile statement
    '''
    print(y_true, y_pred)
    #y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=num_classes)[...,1:])
    y_true_f = y_true[...,1:]#K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=num_classes)[...,1:])
    #y_pred_f = K.flatten(y_pred[...,1:])
    y_pred_f = y_pred[...,1:]
    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)
    return K.mean((2. * intersect / (denom + smooth)))


def dice_coef(y_true, y_pred, smooth=0.001):
    import numpy as np
    y_true_f = K.flatten(y_true[...,1:])
    y_pred_f = K.flatten(y_pred[...,1:])
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    # axes = (1, 2)  # W,H axes of each image
    # intersection = K.sum(K.abs(y_pred_f * y_true_f), axis=axes)
    # mask_sum = K.sum(K.abs(y_true_f)) + K.sum(K.abs(y_pred_f))
    # union = mask_sum - intersection
    # return K.mean(2 * (intersection + smooth)/(mask_sum + smooth))


def tversky(y_true, y_pred, smooth=1, alpha=0.05):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)
