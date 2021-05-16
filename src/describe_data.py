"""
This file read the csv of metadata and compute the summary of the dataset
"""

import pandas as pd
#pd.set_option('display.max_columns', None)

class DataDescription:
    """
    describe the dataset.
    This class is not useful in terms of processing the data

    Args:
        csv_path: path to the csv where there are the metadata of the images.
                  The csv must have this columns: [file_name,class,enviroment,
                  lighting,object_number,background,bg_description,rows,cols,
                  file_size(MB)]
    """
    def __init__(self, csv_path):
        self.csv_data = pd.read_csv(csv_path)

    @property
    def n_class(self):
        return self.csv_data['class'].nunique()
    @property
    def n_images(self):
        return len(self.csv_data)
    @property
    def db_size(self):
        return round(self.csv_data['file_size(MB)'].sum(),3)
    @property
    def img_resolution(self):
        return self.csv_data[['rows','cols']].drop_duplicates()

    def __build_summary_by_class(self):
        classes = self.csv_data["class"].unique()

        class_description = {'object_name':list(),
                             'number_of_objects':list(),
                             'background':list(),
                             'lighting':list(),
                             'number_of_repetions':list(),
                             'number_of_samples':list()}
        # compute each class
        for i in classes:
            class_i = self.csv_data.loc[self.csv_data["class"]==i]
            class_description['object_name'].append(i)
            class_description['lighting'].append('4 lighting variations: indoor-day, indoor-night, outdoor-day, outdoor-night')
            
            number_of_objects = class_i['object_number'].unique().max().item() +1
            class_description['number_of_objects'].append(number_of_objects)

            bg = ';'.join(class_i["bg_description"].unique())
            class_description['background'].append(bg)

            index = ['enviroment','lighting','object_number','background','bg_description']
            pv = class_i[index].pivot_table(index=index, aggfunc='size')
            class_description['number_of_repetions'].append(pv.unique().item())
            class_description['number_of_samples'].append(len(class_i))
        
        return pd.DataFrame(class_description).set_index('object_name', drop=True)

    @property
    def global_summary(self):
        print('GLOBAL TABLE\n')
        print('Description: value\n')
        print(f'Number of classes: {self.n_class}')
        print(f'Number of images: {self.n_images}')
        print(f'Database size: {self.db_size} MB')
        print(f'Image resolution:\n{self.img_resolution}')

    @property
    def summary_by_class(self):
        summary = self.__build_summary_by_class()
        print(summary)