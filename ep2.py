import os

import pandas as pd

from src.data_reader import ObjectDataset
from src.augmenter import Transform, Augmenter
from src.transform.img_transformers import Rgb2Gray, LogTransform, ImageEqualization, Resize
from src.normalize import Normalize, ProcessNormalized


base_path = '/home/suayder/Desktop/visao/data_prep/dataset/data'
augmented_path = '/home/suayder/Desktop/visao/data_prep/dataset/augmented_data1'
normalized_path = '/home/suayder/Desktop/visao/data_prep/dataset/normalized'

trasnformations = Transform.compose([Rgb2Gray(), LogTransform(c=0.3)])


dset = ObjectDataset(base_path, os.path.join(base_path,'metadata.csv'))
classes = dset.get_classes()

df_description = pd.DataFrame(columns=dset.df_csv.columns)
df_description.index.name = dset.df_csv.index.name

for c_name in classes:
    data_augmenter = Augmenter(base_path,
                               os.path.join(base_path,'metadata.csv'),
                               trasnformations)
    
    df_description = pd.concat([df_description,
                                data_augmenter.process_by_class(c_name, save_path = augmented_path)])
    del data_augmenter
df_description.to_csv(os.path.join(augmented_path, 'augmented_metadata.csv'))

## HISTOGRAM EQUALIZATION

#histogram_transform = Transform.compose([ImageEqualization()])
#data_normalizer = Normalize(base_path,os.path.join(base_path,'metadata.csv'),histogram_transform)
# transformed = data_normalizer.show_image('472.JPEG')
exit(0)
#augmented = data_augmenter.process_item('7.jpg')
data_augmenter.process_dataset_and_save(save_path = augmented_path)
#image, _ = data.get_item('7.jpg')
#data.show_item('7.jpg')

## HISTOGRAM EQUALIZATION
histogram_transform = Transform.compose([ImageEqualization(),
                                         Resize(shape=(2048,1536))])
data_normalizer = Normalize(os.path.join(augmented_path, 'data'),
                            os.path.join(augmented_path,'augmented_metadata.csv'),
                            histogram_transform)

#data_normalizer.process_dataset_and_save(normalized_path)


# OPERATIONS IN NORMALIZED DATASET
process_normalized = ProcessNormalized(normalized_path,
                                       os.path.join(augmented_path,'augmented_metadata.csv'))

process_normalized.process_by_class()
