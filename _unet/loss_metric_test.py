import math

import numpy as np
from skimage import io, transform
import matplotlib.pyplot as plt
import segmentation_models as sm
import tensorflow as tf
from sklearn.utils.extmath import cartesian

from _losses import losses, metrics
import keras.backend as K
import cv2


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


def jaccard_loss(y_true, y_pred):
    return 1 - jaccard_score(y_true, y_pred)


def jaccard_score(y_true, y_pred, smooth=0.001):
    import numpy as np
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f + y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)



def weighted_jaccard_score(y_true, y_pred, weights = [0.08392381, 0.97903492, 0.9408015, 0.99623977]):
    # jaccard = 0
    # for i, weight in enumerate(weights):
    #     jaccard += (weight * jaccard_score_class(y_true,y_pred,i))
    # return jaccard / (len(weights) - 1)
    jaccard = 0.0
    n_classes = 0.0
    for i in range(3):
        score = jaccard_score_class(y_true,y_pred,i + 1)
        if score >= tf.constant(0, dtype='float32'):
            jaccard += score
            n_classes += 1
    return jaccard / n_classes


def jaccard_score_class(y_true, y_pred, cl, smooth=0.001):
    import numpy as np
    y_true_f = K.flatten(y_true[...,cl])
    y_pred_f = K.flatten(y_pred[...,cl])
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f + y_pred_f)
    if tf.equal(union, tf.constant(0, dtype='float32')):
        return tf.constant(-1, dtype='float32')
    union = union - intersection
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

def jaccard_loss(y_true, y_pred):
    #return ((1 - jaccard_score_1(y_true, y_pred)) + (1 - jaccard_score_2(y_true, y_pred)) + (1 - jaccard_score_3(y_true, y_pred))) + (1 - jaccard_score(y_true, y_pred))# + categorical_focal_loss(alpha=[0.25,.25,.25,.25])(y_true, y_pred)
    return (1 - weighted_jaccard_score(y_true, y_pred))

def dice_coef(y_true, y_pred, smooth = 0.0001):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_multilabel(y_true, y_pred, numLabels=4):
    dice=0
    for index in range(numLabels):
        dice -= dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index])
    return dice


def tversky_loss(y_true, y_pred):
    alpha = 0.5
    beta = 0.5

    ones = K.ones(K.shape(y_true))
    p0 = y_pred  # proba that voxels are class i
    p1 = ones - y_pred  # proba that voxels are not class i
    g0 = y_true
    g1 = ones - y_true

    num = K.sum(p0 * g0, (0, 1, 2, 3))
    den = num + alpha * K.sum(p0 * g1, (0, 1, 2, 3)) + beta * K.sum(p1 * g0, (0, 1, 2, 3))

    T = K.sum(num / den)  # when summing over classes, T has dynamic range [0 Ncl]

    Ncl = K.cast(K.shape(y_true)[-1], 'float32')
    return Ncl - T


def jaccard_score_all(y_true, y_pred, smooth=0.001):
    import numpy as np
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f + y_pred_f) - intersection
    return (intersection) / (union + smooth)


gt = io.imread(f'../dataset/test/mask/movie_2.mov_0.036667.png.png')
gt = cv2.resize(gt, (512,512), interpolation=cv2.INTER_NEAREST)
# plt.imshow(gt)
# plt.show()
#gt = np.zeros((512,512,4))
gt = tf.keras.utils.to_categorical(gt, num_classes = 4)
gt = np.expand_dims(gt, axis=0)
# plt.imshow(gt[0])
# plt.show()
print(gt.shape)
pr = np.zeros((1,512,512,4), dtype=np.float32)
#pr[:,:,:,1] = 1
# plt.imshow(pr[0])
# plt.show()
gt = np.argmax(gt, axis = -1).astype('float32')
print(gt.shape)
pr = gt

# metric = sm.metrics.IOUScore(class_indexes=[1,2,3])
# print(metric(gt, pr))

print(weighted_hausdorff_distance(512,512,0.001)(gt,pr))
#print(weighted_jaccard_score(gt,pr))
# print(dice_coef_multilabel(gt, pr))
# print(tversky_loss(gt, pr))
#print(jaccard_loss(gt,pr))
#print(jaccard_score_all(gt,pr))
# print(jaccard_score_2(gt,pr))
# print(jaccard_score_3(gt,pr))