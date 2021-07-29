import os
import random
import numpy as np
from skimage.transform import resize
from skimage.exposure import rescale_intensity
from skimage import img_as_uint
import matplotlib.pyplot as plt
from src.utils import read_img, display_bbox,crop_bbox, display
from src.segment import object_segmentation, get_bbox
from src.classifier import SVMClassifier
from src.data_reader import ObjectDataset

#the image will be resized for this size
DEFAULT_SIZE = (500,500)
TRAIN_SPLIT = 0.8
BASE_PATH = '/gdrive/MyDrive/MAC5768-visao_e_processamento_de_imagens/dataset/original_dataset_gray'
CSV_PATH = os.path.join(BASE_PATH, 'metadata.csv')
IMGS_PATH = os.path.join(BASE_PATH,'data')
MASKS_PATH = '/gdrive/MyDrive/MAC5768-visao_e_processamento_de_imagens/dataset/masks'
LIST_MASKS = os.path.join(MASKS_PATH,'manualy_seg.csv')


def preprocess_dataset(image:np.ndarray):

    image = rescale_intensity(image, in_range=(50, 200))
    image = img_as_uint(resize(image, DEFAULT_SIZE))
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
    image = img_as_uint(resize(image, DEFAULT_SIZE))
    #image = (image<127).astype(int)
    return image, img_class

def load_dataset(elements):
    obj_dataset = ObjectDataset(IMGS_PATH, CSV_PATH)

    X = []
    y = []
    for element in elements:
        try:
          image, img_class = read_dset(obj_dataset, element)
        except KeyError:
          continue
        bb = get_bbox(image)
        X.append(crop_bbox(image, bb))
        y.append(img_class)

    return np.array(X),np.array(y)

def train_classifier(train_split):
    """
    args:
        list_images: list of images name to use in training
    """

    #build dataset
    X_train, y_train = load_dataset(train_split)
    svm = SVMClassifier(DEFAULT_SIZE, 2)
    svm.fit(np.array(X_train), np.array(y_train))
    
    return svm

def read_dataset_from_csv():
    """
    this time we use the csv file to have a list of all images that we whant to
    train or predict, it return a list of splited images
    """
    
    names = np.loadtxt(LIST_MASKS, dtype=str, delimiter=',')
    names = np.array(list(map(lambda s: s.replace("'",''), names)))
    random.shuffle(names)
    t_size = int(len(names)*TRAIN_SPLIT)
    train_split = names[:t_size]
    test_split = names[t_size:]
    print("Size of the dataset:", len(names))
    print(f"train split with {t_size} images\ntest split with {(len(names)-t_size)} images")
    return train_split, test_split

def metrics_pipeline():

    #split datset into train and test
    train_split, test_split = read_dataset_from_csv()
  
    #svm = train_classifier(train_split)
    
    segmented_images = []
    x_test, y_test = load_dataset(test_split)

    for element in x_test:
      #print(element)
      #display([element])
      #plt.show()
      image, bbox = seg_pipeline(element, show=False)
      image = crop_bbox(image, bbox)
      segmented_images.append(image)
    #print(len(segmented_images)) 
    #pr = svm.predict(np.expand_dims(segmented_images, axis=0))
    #print(pr)
    #display(image)
    #plt.show()

metrics_pipeline()