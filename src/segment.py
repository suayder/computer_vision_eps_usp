from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import skimage.morphology as morph
from skimage.filters import threshold_otsu, gaussian
from skimage import measure, exposure

def object_segmentation(image:np.ndarray)->np.ndarray:
    """
    args:
        image: numpy array with a single two channel image of rank 2 corresponding to (width, heght)
    return:
        out: binary image with object as 1 and background 0
    """
    image = gaussian(image, sigma=5)

    seed = np.copy(image)
    seed[1:-1, 1:-1] = image.max()
    mask = image
    filled = morph.reconstruction(seed, mask, method='erosion')

    seed = np.copy(image)
    seed[1:-1, 1:-1] = image.min()
    rec = morph.reconstruction(seed, mask, method='dilation')

    out = exposure.rescale_intensity((filled-rec))

    thres = threshold_otsu(out)
    out = out>thres

    out = morph.area_closing(out)
    out = morph.area_opening(out)

    out = measure.label(out,connectivity=2)

    bbox = None
    label = None
    max_area = 0
    for region in measure.regionprops(out):
        # take region with large area
        if region.area > max_area:
            label = region.label
            max_area = region.area
    out = (out==label).astype(int)

    return out

def get_bbox(image:np.ndarray):

    max_area = 0
    bbox = (0,0,image.shape[0], image.shape[1])

    for region in measure.regionprops(image):
        # take region with large area
        if region.area > max_area:
            bbox = region.bbox
            max_area = region.area
    bbox
    # meas = measure.regionprops(image)
    # if len(meas)==0:
    #   return None
    # if len(meas)>1:
    #   meas = meas[0]
    return bbox