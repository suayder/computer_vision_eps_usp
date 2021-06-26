"""
daqui ele chama a função de transformation implementada (que é a equalização da imagem)
e então aplica, salva e faz as operações de extrair protótipo, histograma médio variância do histograma
"""
import os
from skimage import io
from src.augmenter import Transform, Augmenter
from src.utils import build_histogram

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
        """
        if save_path is None:
            save_path = os.path.join(self.base_path, 'normalized')
        os.makedirs(save_path, exist_ok=True)
    
        for name, path in self.paths.items():
            image, obj_class = self.get_item(name)
            transformed = self.tranformations.apply(image, name)

            #save
            class_path = os.path.join(save_path, obj_class)
            if not os.path.exists(class_path):
                os.makedirs(class_path)

            for name, image in transformed.items():
                image_path = os.path.join(class_path, name)
                io.imsave(image_path, image)

    def process_by_class(self, *args):
        """
        process item by class. The idea is almost the same as the function process_dataset_and_save
        but in this case we can filter the images by class.
        The reason of this class is to use to create prototypes by class of each equalized image in the class

        OBS: olhe que talvez tenha alguma função na classe que lê o dataset que já ajude a 
            pegar estas imagens por classe (eu não lembro se tem mesmo)
        """

    def build_prototype(self, img_class: str):
        pass
    
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