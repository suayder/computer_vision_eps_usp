"""
daqui ele chama a função de transformation implementada (que é a equalização da imagem)
e então aplica, salva e faz as operações de extrair protótipo, histograma médio variância do histograma
"""
import os
from skimage import io
import numpy as np
from matplotlib import pyplot as plt
from math import sqrt, ceil
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
            image, obj_class = self.get_item(name, cache=False)
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


    def process_by_class(self):
        """
        process item by class. The idea is almost the same as the function process_dataset_and_save
        but in this case we can filter the images by class.
        The reason of this class is to use to create prototypes by class of each equalized image in the class

        """
        class_names = self.get_classes()
        prototypes = []
        mean_hist = []
        variance = []
        for class_name in class_names:
            mean = self.build_mean_prototype(class_name)
            m_hist, var = self.histogram_stats_by_class(class_name)
            prototypes.append((mean, class_name))
            mean_hist.append((m_hist, class_name))
            variance.append((var,class_name))

       
        # PLOT IMAGES
        total_len = len(prototypes)
        rows = round(sqrt(total_len))
        cols = round(total_len/rows) if round(total_len/rows)*rows>=total_len else ceil(total_len/rows)

        fig,ax = plt.subplots(rows,cols)#, figsize=(18, 16))
        for el in range(total_len):
            img, cl = prototypes[el]
            ax[el//cols,el%cols].imshow(img, cmap='gray')
            ax[el//cols,el%cols].axis('off')
            ax[el//cols,el%cols].set_title(cl)
        
        fig.tight_layout()
        plt.axis('off')
        #plt.show()
        plt.savefig('mean_prototype.jpg')

        #PLOT HISTOGRAMS
        def plot_hist(h, n):
            total_len = len(h)
            rows = round(sqrt(total_len))
            cols = round(total_len/rows) if round(total_len/rows)*rows>=total_len else ceil(total_len/rows)

            fig,ax = plt.subplots(rows,cols)#, figsize=(18, 16))
            for el in range(total_len):
                img, cl = h[el]
                ax[el//cols,el%cols].plot(img)
                ax[el//cols,el%cols].set_title(cl)
                ax[el//cols,el%cols].grid(axis='y')
            
            fig.tight_layout()
            plt.axis('off')
            #plt.show()
            plt.savefig(f'{n}.jpg')

        plot_hist(mean_hist, 'mean_hist')
        plot_hist(variance, 'variance')


    def _get_histogram(self, image):
        """
        in the utils exists a function that return the histogram
        """
        return build_histogram(image)

    def histogram_stats_by_class(self, img_class):
        """
        extract the mean of the histogram to each class

        Return mean of the histogram and the variance
        """
        names = self.get_items_name_by_class(img_class)
        list_hist = []
        for name in names:
            image, _ = self.get_item(name)
            hist = self._get_histogram(image)
            list_hist.append(hist[0])
        arr = np.array(list_hist)
        arr = np.stack(arr[..., np.newaxis], axis=-1)
        mean_hist = arr.mean(axis=-1)
        variance = arr.var(axis=-1)
        
        return mean_hist, variance