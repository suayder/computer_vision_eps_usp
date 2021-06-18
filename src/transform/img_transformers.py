"""
To construc the class must have at least this two methods:
__call__(self, image: np.ndarray) -> return nd array with the transformated image
name(self) -> this is a property, the name of the transformation
"""

import numpy as np
from skimage.color import rgb2gray
from skimage import exposure

class Rgb2Gray(object):
    def __call__(self, image:np.ndarray) -> np.ndarray:
        gray = (rgb2gray(image)*255).astype(np.uint8)
        return gray
    
    @property
    def name(self):
        return 'gray'

class Original(object):
    def __call__(self, image:np.ndarray) -> np.ndarray:
        return image
    
    @property
    def name(self):
        return 'original'

class GradientSum(object):
    """
    descrição do exercício (que está no edisciplinas): Soma de fundo com gradiente de níveis de cinza
    """
    pass

class LogTransform(object):
    """
    descrição (que está no edisciplinas): Logaritmo da imagem
    """
    def __init__(self, c) -> None:
        """
        :param c: constant used in the log transformation
        """
        self.c = c

    def __call__(self, image):
        pass

class ExpTransform(object):
    """
    Image exponential
    """
    pass

class MedianFilter(object):
    """
    filtro da média, não esquecer que se deve implementar a convolução
    """
    pass

class ImageEqualization(object):

    def __call__(self, image:np.ndarray) -> np.ndarray:
        if len(image.shape) != 2:
            image = Rgb2Gray().__call__(image)
        
        equalized_image = exposure.equalize_hist(image)
        return equalized_image#(equalized_image*255).astype(np.uint8)

    @property
    def name(self):
        return 'equalized'
