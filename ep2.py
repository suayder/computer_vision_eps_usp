import os

from src.augmenter import Transform, Augmenter
from src.transform.img_transformers import Rgb2Gray, Original, ImageEqualization
from src.normalize import Normalize

base_path = '/home/David_PRO/MAC5768_Visao_Computacional/dataset/data'

trasnformations = Transform.compose([Rgb2Gray(), Original()])
data_augmenter = Augmenter(base_path,
                           os.path.join(base_path,'metadata.csv'),
                           trasnformations)

augmented = data_augmenter.process_item('472.JPEG')
data_augmenter.process_dataset_and_save()
image, _ = data.get_item('472.JPEG')
data.show_item('472.JPEG')

## HISTOGRAM EQUALIZATION

#histogram_transform = Transform.compose([ImageEqualization()])
#data_normalizer = Normalize(base_path,os.path.join(base_path,'metadata.csv'),histogram_transform)
# transformed = data_normalizer.show_image('472.JPEG')