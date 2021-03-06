"""
To construc the class must have at least this two methods:
__call__(self, image: np.ndarray) -> return nd array with the transformated image
name(self) -> this is a property, the name of the transformation
"""

import numpy as np
from scipy import ndimage
from skimage.color import rgb2gray
from skimage import exposure
from skimage.transform import resize
from skimage.morphology import square
from skimage.filters.rank import mean

class Rgb2Gray(object):
    def __call__(self, image:np.ndarray) -> np.ndarray:
        gray = (rgb2gray(image)*255).astype(np.uint8)
        return gray
    
    @property
    def name(self):
        return 'gray'

class GradientSum(object):
    """
    descrição do exercício (que está no edisciplinas): Soma de fundo com gradiente de níveis de cinza
    """

    def __init__(self, dGrad) -> None:
        """
        :param dGrad: define a direção de aplicação do gradiente de sombras na imagem
            0 - Gradiente na Direção Vertical
            1 - Gradiente na Direção Horizontal
            2 - Gradiente na Direção Diagonal
        """
        self.dGrad = dGrad

    def __call__(self, image:np.ndarray) -> np.ndarray:
        if len(image.shape) != 2:
            image = Rgb2Gray().__call__(image)
        
        imgProcess = np.zeros(image.shape, dtype = "uint8")
        
        if (self.dGrad == 0):
            for lin in range(0,image.shape[0]-1):
                imgProcess[lin,:] = image[lin,:] * float(lin/image.shape[0])

        if (self.dGrad == 1):
            for col in range(0,image.shape[1]-1):
                imgProcess[:, col] = image[:,col] * float(col/image.shape[1])

        if (self.dGrad == 2):
            for lin in range(0,image.shape[0]-1):
                for col in range(0,image.shape[1]-1):
                    imgProcess[lin, col] = image[lin,col] * (float(lin/image.shape[0]) + float(col/image.shape[1]))/2

        return imgProcess
    
    @property
    def name(self):
        return 'GradSum'

 
#https://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.adjust_log

class LogTransform(object):
    """
    descrição (que está no edisciplinas): Logaritmo da imagem
    """
    def __init__(self, c) -> None:
        """
        :param c: constant used in the log transformation
        """
        self.c = c

    def __call__(self, image:np.ndarray) -> np.ndarray:
        if len(image.shape) != 2:
            image = Rgb2Gray().__call__(image)

        log_image = exposure.adjust_log(image, self.c)
        return log_image
    
    @property
    def name(self):
        return 'logarithm'

class ExpTransform(object):
    """
    Image exponential
    """
    def __init__(self, c, gamma) -> None:
        """
        :param c: constant used in the log transformation
        """
        self.c = c
        self.gamma = gamma

    def __call__(self, image:np.ndarray) -> np.ndarray:
        if len(image.shape) != 2:
            image = Rgb2Gray().__call__(image)

        exp_image = exposure.adjust_gamma (image, self.gamma, self.c)
        return exp_image
    
    @property
    def name(self):
        return 'exponential'

class MeanFilter(object):
    """
    Filtro da média implementado usando convolução (funcao pronta)
    """
    def __init__(self, size) -> None:
        """
        :param size: (size x size) kernel 
        """
        self.size = size

    def __call__(self, image:np.ndarray) -> np.ndarray:
        if len(image.shape) != 2:
            image = Rgb2Gray().__call__(image)

        k = square (self.size); #square of 1's size x size

        k = k / (self.size * self.size) #averaging kernel

        mean_image = ndimage.convolve(image, k, mode='constant', cval=0.0)

        return mean_image
    
    @property
    def name(self):
        return 'meanFilter'

class MeanFilter2(object):
    """
    Filtro da média implementado usando a função pronta mean 
    """
    def __init__(self, size) -> None:
        """
        :param size: size of the kernel
        """
        self.size = size

    def __call__(self, image:np.ndarray) -> np.ndarray:
        if len(image.shape) != 2:
            image = Rgb2Gray().__call__(image)

        mean_image = mean (image, square (self.size))

        return mean_image
    
    @property
    def name(self):
        return 'meanFilter2'

class ImageEqualization(object):

    def __call__(self, image:np.ndarray) -> np.ndarray:
        if len(image.shape) != 2:
            image = Rgb2Gray().__call__(image)
        
        equalized_image = exposure.equalize_hist(image)
        return (equalized_image*255).astype(np.uint8)

    @property
    def name(self):
        return 'equalized'

class Resize(object):

    def __init__(self, shape:tuple) -> None:
        """
        args:
            shape: shape of the output image, must be a tuple (n_rows, n_cols)
        """
        assert len(shape)==2
        self.shape = shape
    
    def __call__(self, image:np.ndarray) -> np.ndarray:
        resized = resize(image, self.shape, anti_aliasing=True)
        return (resized*255).astype(np.uint8)

    @property
    def name(self):
        return 'resized'
