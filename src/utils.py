import os
from skimage import io, exposure
import numpy as np

def read_img(path: str, as_gray=False):
    if not os.path.exists(path):
        print(f'Not found file: {path}')
        return None
    return io.imread(path, as_gray=as_gray)

def build_histogram(image: np.ndarray):
    return exposure.histogram(image)