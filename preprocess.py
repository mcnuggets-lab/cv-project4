from keras.utils import to_categorical
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os


TRAIN_FILENAME = "./data/train.txt"
TEST_FILENAME = "./data/test.txt"
NUM_OF_CLASSES = 8
TARGET_SIZE = (240, 320)  #(240, 320)
CROPPING = "random"  # 'center', 'random', None
MIRROR_RATE = 0.5  # probability [0, 1) 0 = no mirroring, 0.5 = 50% chance fliplr


def cropping(arr, crop_size, mode):
    if crop_size[0] > arr.shape[0] or crop_size[1] > arr.shape[1]:
        raise ValueError('crop size {0} is larger than original image size: {1}'.format(crop_size, arr.shape))
    if mode.lower() == 'center':
        x0 = arr.shape[0] // 2 - (crop_size[0] // 2)
        y0 = arr.shape[1] // 2 - (crop_size[1] // 2)
    elif mode.lower() == 'random':
        x_diff = arr.shape[0] - crop_size[0]
        y_diff = arr.shape[1] - crop_size[1]
        x0 = np.random.randint(0, max(x_diff - 1, 0)) if x_diff > 0 else 0
        y0 = np.random.randint(0, max(y_diff - 1, 0)) if y_diff > 0 else 0
    else:
        raise NotImplementedError
    if arr.ndim == 3:
        return arr[x0:x0 + crop_size[0], y0:y0 + crop_size[1], :]
    elif arr.ndim == 2:
        return arr[x0:x0 + crop_size[0], y0:y0 + crop_size[1]]
    else:
        raise NotImplementedError


def train_generator(train_filename, crop=CROPPING, mirror=MIRROR_RATE):
    df = pd.read_csv(train_filename, dtype={0: np.str})
    if crop is not None:
        df = df[(df['width'] >= TARGET_SIZE[1]) & (df['height'] >= TARGET_SIZE[0])]
    while True:
        for row in df.iterrows():
            # prepare image
            img_filename = row[1]["filename"]
            img_path = os.path.join(".", "data", "images", "{}.jpg".format(img_filename))
            raw_img = image.load_img(img_path)
            raw_img = image.img_to_array(raw_img)
            if crop is not None and raw_img.shape[:2] != TARGET_SIZE:
                raw_img = cropping(raw_img, crop_size=TARGET_SIZE, mode=crop)
            is_random_flip = np.random.rand() < mirror
            if is_random_flip:
                raw_img = np.fliplr(raw_img)
            img = preprocess_input(raw_img)

            # prepare label
            label_path = os.path.join(".", "data", "labels", "{}.regions.txt".format(img_filename))
            label = np.loadtxt(label_path)
            if crop is not None and label.shape[:2] != TARGET_SIZE:
                label = cropping(label, crop_size=TARGET_SIZE, mode=crop)
            if is_random_flip:
                label = np.fliplr(label)
            label_onehot = to_categorical(label, num_classes=NUM_OF_CLASSES)
            yield np.expand_dims(img, axis=0), np.expand_dims(label_onehot, axis=0)


def get_train_set(train_filename):
    df = pd.read_csv(train_filename, dtype={0: np.str})
    imgs, labels = [], []
    for row in df.iterrows():
        # prepare image
        img_filename = row[1]["filename"]
        img_path = os.path.join(".", "data", "images", "{}.jpg".format(img_filename))
        raw_img = image.load_img(img_path, target_size=TARGET_SIZE)
        raw_img = image.img_to_array(raw_img)
        img = preprocess_input(raw_img)
        imgs.append(img)

        # prepare label
        label_path = os.path.join(".", "data", "labels", "{}.regions.txt".format(img_filename))
        label = np.loadtxt(label_path)
        label_onehot = to_categorical(label, num_classes=NUM_OF_CLASSES)
        labels.append(label_onehot)

    imgs = np.array(imgs)
    return imgs, labels


def preprocess_image(index):
    img_path = os.path.join(".", "data", "images", "{}.jpg".format(index))
    raw_img = image.load_img(img_path, target_size=TARGET_SIZE)
    raw_img = image.img_to_array(raw_img)
    img = preprocess_input(raw_img)
    return img


if __name__ == "__main__":
    gen = train_generator(TRAIN_FILENAME)
    train_img, train_label = next(gen)
    plt.imshow((train_img[0] - np.min(train_img[0])) / 255)