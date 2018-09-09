import numpy as np
from keras.utils import to_categorical

from preprocess import NUM_OF_CLASSES

def per_pixel_accuracy(pred, target):
    target_categorical = np.argmax(target, axis=-1)
    return np.sum(np.argmax(pred, axis=-1) == target_categorical) / target_categorical.size


def mean_per_class_accuracy(pred, target):
    pred = to_categorical(np.argmax(pred, axis=-1), num_classes=NUM_OF_CLASSES)
    accuracies = []
    for i in range(target.shape[-1]):
        gt_sum = target[:, :, i].sum()
        if gt_sum > 0:
            accuracies.append(np.sum(np.logical_and(target[:, :, i], pred[:, :, i])) / gt_sum)
    return np.mean(accuracies)


def mean_IoU(pred, target):
    pred = to_categorical(np.argmax(pred, axis=-1), num_classes=NUM_OF_CLASSES)
    accuracies = []
    for i in range(target.shape[-1]):
        gt_sum = target[:, :, i].sum()
        pred_sum = pred[:, :, i].sum()
        if gt_sum == 0 or pred_sum == 0:
            continue
        intersect_sum = np.sum(np.logical_and(target[:, :, i], pred[:, :, i]))
        accuracies.append(intersect_sum / (gt_sum + pred_sum - intersect_sum))
    return np.mean(accuracies)


def frequency_weighted_IoU(pred, target):
    pred = to_categorical(np.argmax(pred, axis=-1), num_classes=NUM_OF_CLASSES)
    accuracies = []
    for i in range(target.shape[-1]):
        gt_sum = target[:, :, i].sum()
        pred_sum = pred[:, :, i].sum()
        if gt_sum == 0 or pred_sum == 0:
            continue
        intersect_sum = np.sum(np.logical_and(target[:, :, i], pred[:, :, i]))
        accuracies.append(intersect_sum * gt_sum / (gt_sum + pred_sum - intersect_sum))
    return np.sum(accuracies) / (target.shape[0] * target.shape[1])
