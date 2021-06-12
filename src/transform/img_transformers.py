"""
To construc the class must have at least this two methods:
__call__(self, image: np.ndarray) -> return nd array with the transformated image
name(self) -> this is a property, the name of the transformation
"""

import numpy as np
from skimage.color import rgb2gray

class Rgb2Gray(object):
    def __call__(self, image) -> np.ndarray:
        gray = (rgb2gray(image)*255).astype(np.uint8)
        return gray
    
    @property
    def name(self):
        return 'gray'

class Original(object):
    def __call__(self, image) -> np.ndarray:
        return image
    
    @property
    def name(self):
        return 'original'
