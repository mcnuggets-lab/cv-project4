# force cpu usage
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from keras.models import load_model

import numpy as np
import cv2

from utils import BilinearUpSampling2D
from preprocess import preprocess_image, get_train_set, TRAIN_FILENAME, TEST_FILENAME
from visualizer import visualize
from metrics import *


# TRAIN_FILENAME = "./data/train.txt"
# TEST_FILENAME = "./data/test.txt"

model = load_model("./models/fcn8_v1.h5", custom_objects={'BilinearUpSampling2D': BilinearUpSampling2D})

# index = "6000315"
# img_preprocessed = preprocess_image(index)
# pred = model.predict(np.expand_dims(img_preprocessed, axis=0))
# fig = visualize(index, pred[0])

test_images, test_labels = get_train_set(TEST_FILENAME, crop=None, mirror=0)

accuracies = []
for i in range(len(test_labels)):
    pred = model.predict(np.expand_dims(test_images[i], axis=0))
    resized_pred = cv2.resize(pred[0], (test_labels[i].shape[1], test_labels[i].shape[0]))
    acc = per_pixel_accuracy(resized_pred, test_labels[i])
    print(i, acc)
    accuracies.append(acc)
mean_test_acc = np.mean(accuracies)
print("Mean test accuracy: {}".format(mean_test_acc))


