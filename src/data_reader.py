"""
This script loads the dataset and operate with it
"""
import os
from random import sample
from skimage import io
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import math

class ObjectDataset:
    """
    Read the images from its base path and the csv of description
    """

    def __init__(self, base_path:str, csv_path:str) -> None:
        """
        args:
            base_path: Path where the images are
            csv_path: path to the csv of descriptions 
        """
        self.base_path = base_path
        self.df_csv = pd.read_csv(csv_path, index_col=0)
        self.paths = self.__buid_all_dirs()
        self.images = {}

    def __read_img(self, path):
        if not os.path.exists(path):
            print(f'Not found file: {path}')
            return None
        return io.imread(path)

    def __buid_all_dirs(self) -> dict:
        
        paths = {}
        for i,row in self.df_csv.iterrows():
            f_name = i
            f_class = row['class']
            paths[f_name] = os.path.join(self.base_path,f_class, f_name)
        return paths

    def get_item_description(self, img_name:str, features=None) -> str:
        
        if features is None:
            img_class = self.df_csv['class'].loc[img_name]
            lighting = self.df_csv['lighting'].loc[img_name]
            bg = self.df_csv['bg_description'].loc[img_name]
            desc = f'name: {img_name}, class:{img_class}\n\
                    lighting condiction: {lighting}, background: {bg}'
        
        else:
            desc = ''
            for i in features:
                if i in self.df_csv.columns:
                    ft = self.df_csv[i].loc[img_name]
                    desc+=f'{i}: {ft} '

        return desc

    def get_item(self,img_name:str) -> tuple:
        """
        load images on demand
        """
        img_class = self.df_csv['class'].loc[img_name]
        if img_name in self.images:
            img_arr = self.images[img_name]
        else:
            img_arr = self.__read_img(self.paths[img_name])
            self.images[img_name] = img_arr

        return img_arr, img_class

    def show_item(self, img_name:str) -> None:
        img_arr, _ = self.get_item(img_name)
        decription = self.get_item_description(img_name)
        plt.title(decription)
        io.imshow(img_arr)
        plt.show()

    def get_dataset(self) -> list:
        dataset = []
        for name, path in self.paths.items():
            img, obj_class = self.get_item(name)
            if img is not None:
                dataset.append((img, obj_class))
        return dataset

    def show_random_sample(self, samples_by_class:int):
        dataset_sample, names = self.get_random_sample(samples_by_class)
        rows = len(names)//samples_by_class
        cols = samples_by_class
        if rows>10 or cols>15:
            total_len = rows*cols
            rows = round(math.sqrt(total_len))
            cols = round(total_len/rows)

        figure = plt.figure()
        axes = []
        for el in range(len(names)):
            img, cl = dataset_sample[el]
            ax = figure.add_subplot(rows, cols, el+1)
            ax.axis('off')
            axes.append(ax)
            subplot_title=(cl)
            axes[-1].set_title(subplot_title)
            io.imshow(img)
        figure.tight_layout()
        plt.show()

    def show_random_sample_on_colab(self, samples_by_class:int):
        dataset_sample, names = self.get_random_sample(samples_by_class)
        rows = len(names)//samples_by_class
        cols = samples_by_class
        if rows>10 or cols>15:
            total_len = rows*cols
            rows = round(math.sqrt(total_len))
            cols = round(total_len/rows) if round(total_len/rows)*rows>=total_len else math.ceil(total_len/rows)

        fig,ax = plt.subplots(rows,cols, figsize=(18, 16))
        for el in range(len(names)):
            img, cl = dataset_sample[el]
            ax[el//cols,el%cols].imshow(img)
            ax[el//cols,el%cols].axis('off')
            ax[el//cols,el%cols].set_title(cl)
        fig.tight_layout()
        plt.axis('off')
        plt.show()

             
    def get_random_sample(self, samples_by_class:int)->tuple:
        classes = self.df_csv['class'].unique()
        dataset_sample = []
        names = []
        for c in classes:
            df_class = self.df_csv.loc[self.df_csv['class']==c].index.values.tolist()
            for element in sample(df_class, samples_by_class):
                img, _ = self.get_item(element)
                dataset_sample.append((img,c))
                names.append(element)
        return dataset_sample, names