import os
import random
import numpy as np
from skimage.transform import resize
from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt

from src.utils import read_img, display_bbox,crop_bbox, display
from src.segment import object_segmentation, get_bbox
from src.classifier import SVMClassifier
from src.data_reader import ObjectDataset

#the image will be resized for this size
DEFAULT_SIZE = (500,500)
TRAIN_SPLIT = 0.8
BASE_PATH = '/home/suayder/Desktop/visao/data_prep/dataset/augmented_data'
CSV_PATH = os.path.join(BASE_PATH, 'augmented_metadata.csv')
IMGS_PATH = os.path.join(BASE_PATH,'data')


def preprocess_dataset(image:np.ndarray):

    image = rescale_intensity(image, in_range=(50, 200))
    image = resize(image, DEFAULT_SIZE, anti_aliasing=True)
    return image


def seg_pipeline(image, is_binary=False, show=True):
    """
    args:
        image: can be the path to the image or a np array corresponding to the images
    """

    if isinstance(image,str):
        image = read_img(image, as_gray=True)
    
    image = preprocess_dataset(image)
    image = object_segmentation(image) if not is_binary else image
    bbox = get_bbox(image)

    if show:
        display_bbox(image, bbox)
        plt.show()

    return image, bbox

def read_dset(obj_dataset, element):

    image, img_class = obj_dataset.get_item(element,cache=False)
    image = resize(image, DEFAULT_SIZE, anti_aliasing=True)
    return image, img_class


def train_classifier(train_split):
    """
    args:
        list_images: list of images name to use in training
    """

    #build dataset
    obj_dataset = ObjectDataset(IMGS_PATH, CSV_PATH)

    X_train = []
    y_train = []
    for element in train_split:
        image, img_class = read_dset(obj_dataset, element)
        X_train.append(image)
        y_train.append(img_class)
    
    svm = SVMClassifier(DEFAULT_SIZE, 2)
    svm.fit(np.array(X_train), np.array(y_train))
    
    return svm

def read_dataset_from_csv():
    """
    this time we use the csv file to have a list of all images that we whant to
    train or predict, it return a list of splited images
    """
    
    names = np.loadtxt("sample.csv", dtype=str)
    random.shuffle(names)
    t_size = int(len(names)*TRAIN_SPLIT)
    train_split = names[:t_size]
    test_split = names[t_size:]
    print("Size of the dataset:", len(names))
    print(f"train split with {t_size} images\ntest split with {(len(names)-t_size)} images")
    return train_split, test_split

def main_pipeline():

    path = os.path.join(IMGS_PATH,'medicine','gray_10.jpg')
    path1 = os.path.join(IMGS_PATH, 'toothbrush', 'gray_15.jpg')
    
    path = ['gray_10.jpg', 'gray_15.jpg']
    svm = train_classifier(path)
    
    image, bbox = seg_pipeline(path1, show=False)
    image = crop_bbox(image, bbox)
    
    pr = svm.predict(np.expand_dims(image, axis=0))
    print(pr)
    #display(image)
    #plt.show()

main_pipeline()