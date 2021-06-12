import os
import math
import numpy as np
from skimage import io
from matplotlib import pyplot as plt
from src.data_reader import ObjectDataset

class Transform:

    @classmethod
    def compose(cls, transformations:list):
        """
        args:
            transformations: transformations to be applied to each image.
                             See module transform.image_transform
        """
        for transf in transformations:
            assert isinstance(transf, object)
        return cls(transformations)
    
    def __init__(self, transformations) -> None:
        self.transformations = transformations

    def apply(self, img:np.ndarray, img_name:str) -> dict:
        """
        apply all the transformations to a single image
        return:
         dict {key = transfomation_image_name, value=image_array}
        """
        transformed = {}
        for transform in self.transformations:
            comp_name = transform.name+"_"+img_name
            transformed[comp_name] = transform.__call__(img)

        return transformed

class Augmenter(ObjectDataset):
    def __init__(self, base_path:str, csv_path:str, transformations:Transform) -> None:
        """
        args:
            base_path: Path where the images are
            csv_path: path to the csv of descriptions
            transformations: class Transform where apply the transformations in the image
        """
        super().__init__(base_path, csv_path)
        assert isinstance(transformations, Transform)
        self.tranformations = transformations
    
    def process_item(self, img_name:str) -> dict:
        """
        Pass the name of the image and get a dict with augmented images
        """
        image, _ = self.get_item(img_name)

        return self.tranformations.apply(image, img_name)

    def show_augmented_item(self, img_name:str) -> None:
        """
        img_name: image name to be augmented and ploted
        """

        images = self.process_item(img_name)
        total_len = len(images)
        rows = round(math.sqrt(total_len))
        cols = round(total_len/rows)
        figure = plt.figure()
        axes = []

        for el, image_name in enumerate(images.keys()):
            
            img = images[image_name]
            ax = figure.add_subplot(rows, cols, el+1)
            ax.axis('off')
            axes.append(ax)
            subplot_title=image_name.split('_')[0]
            axes[-1].set_title(subplot_title)
            io.imshow(img)
        figure.tight_layout()
        plt.show()

    
    def process_dataset_and_save(self, save_path:str = None):
        """
        save_path: path with the base_dir to save,if none a folder with
                   name augmented will be created in the base_path
        """

        if save_path is None:
            save_path = os.path.join(self.base_path, 'augmented')
            os.makedirs(save_path)

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