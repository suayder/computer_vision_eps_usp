"""
daqui ele chama a função de transformation implementada (que é a equalização da imagem)
e então aplica, salva e faz as operações de extrair protótipo, histograma médio variância do histograma
"""
import os
from skimage import io
import numpy as np
from src.augmenter import Transform, Augmenter
from src.data_reader import ObjectDataset
from src.utils import build_histogram, show_array

class Normalize(Augmenter):
    """
    Normalize the images applying a histogram equalization operation
    """
    def __init__(self, base_path:str, csv_path:str, transformations:Transform) -> None:
        """
        args:
            base_path: Path where the images are. THE IMAGES MUST BE IN GRAYSCALE.
            csv_path: path to the csv of descriptions
            transformations: class Transform where apply the transformations in the image
        """
        super().__init__(base_path, csv_path, transformations)

    def process_item(self, img_name:str) -> dict:
        """
        process histogram for a single image
        """
        return super().process_item(img_name)

    def show_image(self, img_name:str) -> None:
        super().show_augmented_item(img_name)

    def process_dataset_and_save(self, save_path:str = None):
        """
        this function process all the dataset and save it.
        equalize the image and resize it
        """
        if save_path is None:
            save_path = os.path.join(self.base_path, 'normalized')
        os.makedirs(save_path, exist_ok=True)
    
        for name, path in self.paths.items():
            image, obj_class = self.get_item(name)
            transformed_image, name = self.tranformations.apply_sequential(image, name)
            name = '_'.join(name.split('_')[-2:])

            #save
            class_path = os.path.join(save_path, obj_class)
            if not os.path.exists(class_path):
                os.makedirs(class_path)

            image_path = os.path.join(class_path, name)
            io.imsave(image_path, transformed_image)

class ProcessNormalized(ObjectDataset):
    """
    This class process normalized dataset generating statistics of the dataset
    """

    def __init__(self, base_path:str, csv_path:str) -> None:
        """
        args:
            base_path: Path where the images are. THE IMAGES MUST BE IN GRAYSCALE.
            csv_path: path to the csv of descriptions
        """
        super().__init__(base_path, csv_path)

    def build_mean_prototype(self, img_class: str):

        names = self.get_items_name_by_class(img_class)
        list_images = []
        for name in names:
            image, _ = self.get_item(name)
            list_images.append(image)
        arr = np.array(list_images)
        arr = np.stack(arr[..., np.newaxis], axis=-1)
        mean_arr = arr.mean(axis=-1)
        mean_arr = np.rint(mean_arr).astype(np.uint8)
        return mean_arr

    def process_by_class(self, save=True):
        """
        process item by class. The idea is almost the same as the function process_dataset_and_save
        but in this case we can filter the images by class.
        The reason of this class is to use to create prototypes by class of each equalized image in the class

        args:
            save: if true the image will be saved, else the image will only be showed
        """
        class_names = self.get_classes()
        for class_name in class_names:
            mean = self.build_mean_prototype(class_name)
            if save:
                io.imsave((self.base_path+f'mean_prot_{class_name}.png'), mean)
            else:
                show_array(mean)

    def get_histogram(self):
        """
        in the utils exists a function that return the histogram
        """
        pass

    def mean_histogram_by_class(self):
        """
        extract the mean of the histogram to each class
        """
        pass

    def class_hist_variance(self):
        """
        variance of the histogram
        """
        pass

# OBS: implementar somente o metodo que calcula as médias variancias, etc e o "retorno" dele seja mostrar
#       uma imagem com estes graficos é o suficiente