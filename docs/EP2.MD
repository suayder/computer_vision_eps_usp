# [EP2] MAC0417/5768 - Visão e Processamento de Imagens (2021)

### Authors:
 - Suayder Milhomem
 - David Oliveira
 - Daniel Hotta


## Simple description

This notebook uses the the classes that we have created in order to describe and read our dataset of objects.

- The libs you can find in this [repository](https://github.com/suayder/computer_vision_eps_usp.git) in a branch named EP2

- The full dataset can be found [here](https://drive.google.com/drive/folders/1GJD9P-zUVVOHRNSenVLbm_XS1joCXYm-?usp=sharing).

- The augmented dataset can be found [here](https://drive.google.com/drive/folders/12VkujKaxqRLSw0WmiearpAcSVmWcGUaL?usp=sharing)

- The normalized dataset can be found [here](https://drive.google.com/drive/folders/1GiaHAkYCSkgVvktgkbBAYJIR9ETHqxpw?usp=sharing)

based on the goals of the EP2 we show each result in the the next sections.

## Image augmentation:

The dataset was augmented using the operations:

- [RGB2gray](https://github.com/suayder/computer_vision_eps_usp/blob/6bb0e46e5ddc7db17ac501f61fcb1701e96ad4a4/src/transform/img_transformers.py#L15)
- [GradientSum](https://github.com/suayder/computer_vision_eps_usp/blob/6bb0e46e5ddc7db17ac501f61fcb1701e96ad4a4/src/transform/img_transformers.py#L24)
- [LogTransform](https://github.com/suayder/computer_vision_eps_usp/blob/6bb0e46e5ddc7db17ac501f61fcb1701e96ad4a4/src/transform/img_transformers.py#L66)
- [ExpTransform](https://github.com/suayder/computer_vision_eps_usp/blob/6bb0e46e5ddc7db17ac501f61fcb1701e96ad4a4/src/transform/img_transformers.py#L87)
- [MeanFilter](https://github.com/suayder/computer_vision_eps_usp/blob/6bb0e46e5ddc7db17ac501f61fcb1701e96ad4a4/src/transform/img_transformers.py#L109)

Just to understand the code flow:

1. First the class [Transform](https://github.com/suayder/computer_vision_eps_usp/blob/6bb0e46e5ddc7db17ac501f61fcb1701e96ad4a4/src/augmenter.py#L10) is instatiated passing the desired transformations functions.

2. Then the class [Augmenter](https://github.com/suayder/computer_vision_eps_usp/blob/6bb0e46e5ddc7db17ac501f61fcb1701e96ad4a4/src/augmenter.py#L45) is instantiated passing the object of Transform, and then we can call the methods, to transform. *In our specific case we called [process_by_class](https://github.com/suayder/computer_vision_eps_usp/blob/6bb0e46e5ddc7db17ac501f61fcb1701e96ad4a4/src/augmenter.py#L146)*

# Normalization

Basicaly to normalize the dataset we run the same flow of the code used to augment but instead we have passed the [ImageEqualization](https://github.com/suayder/computer_vision_eps_usp/blob/361f487da8dce0d0447aded7adb57baeafbf84d4/src/transform/img_transformers.py#L157) and [Resize](https://github.com/suayder/computer_vision_eps_usp/blob/361f487da8dce0d0447aded7adb57baeafbf84d4/src/transform/img_transformers.py#L170).

The idea is to resize the image first to keep all of them the same size and then normalize the histogram by equalization.


# Some images after transformed

## Augmentation

![](images/augmented1.png)

![](images/augmented2.png)

![](images/augmented3.png)