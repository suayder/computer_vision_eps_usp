import os

from src.augmenter import Transform, Augmenter
from src.transform.img_transformers import Rgb2Gray, LogTransform, ImageEqualization
from src.normalize import Normalize

base_path = '/home/suayder/Desktop/visao/data_prep/dataset/data'
augmented_path = '/home/suayder/Desktop/visao/data_prep/dataset/augmented_data'
normalized_path = '/home/suayder/Desktop/visao/data_prep/dataset/normalized_data'

trasnformations = Transform.compose([Rgb2Gray(), LogTransform(c=0.3)])
data_augmenter = Augmenter(base_path,
                           os.path.join(base_path,'metadata.csv'),
                           trasnformations)

#augmented = data_augmenter.process_item('7.jpg')
#data_augmenter.process_dataset_and_save(save_path = augmented_path)
#image, _ = data.get_item('7.jpg')
#data.show_item('7.jpg')


## HISTOGRAM EQUALIZATION
histogram_transform = Transform.compose([ImageEqualization()])
data_normalizer = Normalize(os.path.join(augmented_path, 'data'),
                            os.path.join(augmented_path,'augmented_metadata.csv'),
                            histogram_transform)

data_normalizer.process_dataset_and_save(normalized_path)