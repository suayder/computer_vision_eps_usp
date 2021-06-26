import os
from skimage import io, exposure
import numpy as np
from matplotlib import pyplot as plt

def read_img(path: str, as_gray=False):
    if not os.path.exists(path):
        print(f'Not found file: {path}')
        return None
    return io.imread(path, as_gray=as_gray)

def show_array(image: np.array):
    plt.imshow(image,cmap=plt.cm.gray)
    plt.show()

def build_histogram(image: np.ndarray):
    return exposure.histogram(image)