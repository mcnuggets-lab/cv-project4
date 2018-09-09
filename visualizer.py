import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from keras.models import load_model
from keras.utils import plot_model

from preprocess import preprocess_image, NUM_OF_CLASSES
from utils import BilinearUpSampling2D

COLOR_DICT = np.array([
    [135, 206, 235],  # sky: sky-blue
    [78, 110, 56],    # tree: tree-green
    [151, 126, 102],  # road: grey-brown
    [0, 255, 0],      # grass: green
    [0, 0, 255],      # water: blue
    [255, 165, 0],    # building: orange
    [160, 32, 240],   # mountain: purple
    [255, 0, 0]       # foreground objects: red
])

CLASSES = ["sky", "tree", "road", "grass", "water", "building", "mountain", "foreground object"]
LEGEND_DICT = {CLASSES[i]: COLOR_DICT[i] / 255 for i in range(len(COLOR_DICT))}


def segmentation_visualizer(seg):
    img = np.zeros((*seg.shape, 3))
    for i in range(NUM_OF_CLASSES):
        img[seg == i] = COLOR_DICT[i]
    return img / 255


def visualize(file_index, prediction):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.axis("off")
    ax2.axis("off")
    ax3.axis("off")

    img_path = os.path.join(".", "data", "images", "{}.jpg".format(file_index))
    image = plt.imread(img_path)
    ax1.imshow(image)
    ax1.set_title("Original Image")

    img_pred = np.argmax(prediction, axis=-1)
    ax2.imshow(segmentation_visualizer(img_pred))
    ax2.set_title("Prediction")

    label_path = os.path.join(".", "data", "labels", "{}.regions.txt".format(file_index))
    img_truth = np.loadtxt(label_path)
    ax3.imshow(segmentation_visualizer(img_truth))
    ax3.set_title("Ground Truth")

    fig.show()

    return fig


def visualize_with_legend(file_index, prediction):
    ax1 = plt.subplot2grid((2, 3), (0, 0))
    ax2 = plt.subplot2grid((2, 3), (0, 1))
    ax3 = plt.subplot2grid((2, 3), (0, 2))
    ax4 = plt.subplot2grid((2, 3), (1, 0), colspan=3)
    ax1.axis("off")
    ax2.axis("off")
    ax3.axis("off")
    ax4.axis("off")

    img_path = os.path.join(".", "data", "images", "{}.jpg".format(file_index))
    image = plt.imread(img_path)
    ax1.imshow(image)
    ax1.set_title("Original Image")

    img_pred = np.argmax(prediction, axis=-1)
    ax2.imshow(segmentation_visualizer(img_pred))
    ax2.set_title("Prediction")

    label_path = os.path.join(".", "data", "labels", "{}.regions.txt".format(file_index))
    img_truth = np.loadtxt(label_path)
    ax3.imshow(segmentation_visualizer(img_truth))
    ax3.set_title("Ground Truth")

    patchList = []
    for key in LEGEND_DICT:
        data_key = mpatches.Patch(color=LEGEND_DICT[key], label=key)
        patchList.append(data_key)
    ax4.legend(handles=patchList, ncol=len(LEGEND_DICT) // 2, loc=9)

    plt.show()


def visualize_all_models(file_index, predictions):
    ax1 = plt.subplot2grid((3, 3), (0, 0))
    ax2 = plt.subplot2grid((3, 3), (0, 1))
    ax3 = plt.subplot2grid((3, 3), (0, 2))
    ax4 = plt.subplot2grid((3, 3), (1, 0))
    ax5 = plt.subplot2grid((3, 3), (1, 1))
    ax6 = plt.subplot2grid((3, 3), (2, 0), colspan=3)
    ax1.axis("off")
    ax2.axis("off")
    ax3.axis("off")
    ax4.axis("off")
    ax5.axis("off")
    ax6.axis("off")

    img_path = os.path.join(".", "data", "images", "{}.jpg".format(file_index))
    image = plt.imread(img_path)
    ax4.imshow(image)
    ax4.set_title("Original Image")

    img_pred_32 = np.argmax(predictions[0], axis=-1)
    ax1.imshow(segmentation_visualizer(img_pred_32))
    ax1.set_title("FCN-32s")

    img_pred_16 = np.argmax(predictions[1], axis=-1)
    ax2.imshow(segmentation_visualizer(img_pred_16))
    ax2.set_title("FCN-16s")

    img_pred_8 = np.argmax(predictions[2], axis=-1)
    ax3.imshow(segmentation_visualizer(img_pred_8))
    ax3.set_title("FCN-8s")

    label_path = os.path.join(".", "data", "labels", "{}.regions.txt".format(file_index))
    img_truth = np.loadtxt(label_path)
    ax5.imshow(segmentation_visualizer(img_truth))
    ax5.set_title("Ground Truth")

    patchList = []
    for key in LEGEND_DICT:
        data_key = mpatches.Patch(color=LEGEND_DICT[key], label=key)
        patchList.append(data_key)
    ax6.legend(handles=patchList, ncol=len(LEGEND_DICT) // 2, loc=9)

    plt.show()


if __name__ == "__main__":
    index = "6000071" # "3002020"
    img = plt.imread("./data/images/{}.jpg".format(index))
    img_preprocessed = preprocess_image(index)
    # pred = np.random.rand(1, 240, 320, 8)

    ALL_MODELS_MODE = True
    if ALL_MODELS_MODE:
        predictions = []
        for stride in [32, 16, 8]:
            model = load_model("./models/fcn{}_v1.h5".format(stride),
                               custom_objects={'BilinearUpSampling2D': BilinearUpSampling2D})
            pred = model.predict(np.expand_dims(img_preprocessed, axis=0))
            resized_pred = cv2.resize(pred[0], (img.shape[1], img.shape[0]))
            predictions.append(resized_pred)
        visualize_all_models(index, predictions)
    else:
        model = load_model("./models/fcn8_v1.h5", custom_objects={'BilinearUpSampling2D': BilinearUpSampling2D})
        # plot_model(model, to_file='./plots/fcn8_architecture.png')
        pred = model.predict(np.expand_dims(img_preprocessed, axis=0))
        resized_pred = cv2.resize(pred[0], (img.shape[1], img.shape[0]))
        # fig = visualize(index, resized_pred)
        visualize_with_legend(index, resized_pred)




