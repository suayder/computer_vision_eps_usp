import os
from skimage import io, exposure
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

def read_img(path: str, as_gray=False):
    if not os.path.exists(path):
        print(f'Not found file: {path}')
        return None
    return io.imread(path, as_gray=as_gray)

def show_array(image: np.array):
    plt.imshow(image,cmap=plt.cm.gray)
    plt.show()

def display(imgs, n_cols=1, titles=None, figsize=(15, 10), **kwargs):
    if isinstance(imgs, dict):
        imgs = list(imgs.values())
        titles = list(imgs.keys())
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]
    if titles is None:
        titles = len(imgs)*[None]
    assert len(titles) == len(imgs)
        
    n_rows = np.ceil(len(imgs)/n_cols)
    plt.figure(figsize=figsize)
    for i, (img, title) in enumerate(zip(imgs, titles)):
        plt.subplot(n_cols, n_rows, i+1)
        plt.imshow(img)
        plt.title(title)
        plt.grid(False)
        plt.axis(False)

def display_bbox(image:np.ndarray, bbox):
    """
    args:
        image: image to draw bounding box over
        bbox: iterable of components (min_r, minc, maxr, maxc)
    """
    if bbox is not None:
        minr, minc, maxr, maxc = bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                    fill=False, edgecolor='red', linewidth=2)

        fig, ax = plt.subplots()
        ax.add_patch(rect)
        ax.imshow(image, cmap=plt.cm.gray)
    else:
        fig, ax = plt.subplots()
        ax.imshow(image, cmap=plt.cm.gray)

def crop_bbox(image:np.ndarray, bbox):
    """
    args:
        image: original image
        bbox: iterable of components (minr, minc, maxr, maxc)
    """

    if bbox is not None:
        minr, minc, maxr, maxc = bbox
    else:
        minr, minc, maxr, maxc = 0,0,image.shape[0], image.shape[1]
    return image[minr:maxr, minc:maxc]

def build_histogram(image: np.ndarray):
    return exposure.histogram(image)