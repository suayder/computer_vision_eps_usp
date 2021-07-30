# [EP3] MAC0417/5768 - Visão e Processamento de Imagens (2021)

### Authors:
 - Suayder Milhomem
 - David Oliveira
 - Daniel Hotta


## Simple description

This notebook uses the the classes that we have created in order to describe and read our dataset of objects.

- The libs can be found in this [repository](https://github.com/suayder/computer_vision_eps_usp.git) in a branch named EP3

- The full dataset of images can be found [here](https://drive.google.com/drive/folders/1GJD9P-zUVVOHRNSenVLbm_XS1joCXYm-?usp=sharing).

- The augmented dataset can be found [here](https://drive.google.com/drive/folders/12VkujKaxqRLSw0WmiearpAcSVmWcGUaL?usp=sharing)

- The normalized dataset can be found [here](https://drive.google.com/drive/folders/1GiaHAkYCSkgVvktgkbBAYJIR9ETHqxpw?usp=sharing)

- The manual ground-truth cab be found [here](https://drive.google.com/drive/folders/1lKkDUXQHloZy4YMkQMyr7Qx7NTsfO5wq?usp=sharing)


Based on the goals of the EP3 we listed each result in separate sections as follows:

# 3.1 Image segmentation:

The dataset was segmented using the following code:

- [metrics](https://github.com/suayder/computer_vision_eps_usp/blob/EP3/src/metrics.py)

- [segmentation](https://github.com/suayder/computer_vision_eps_usp/blob/EP3/src/segment.py)

A brief explanation of the code flow we adopted:

1. a

2. a


## Manual Segmentation Samples

![](images/manual_segmentation.png)

![](images/manual_segmentation2.png)

## Automatic Segmentation Samples


## Performance


# 3.2 Classification

The dataset was classified using the following code:

- [metrics](https://github.com/suayder/computer_vision_eps_usp/blob/EP3/src/metrics.py)

- [classifier](https://github.com/suayder/computer_vision_eps_usp/blob/EP3/src/classifier.py)

A brief explanation of the code flow we adopted:

1. a

2. a

## Classification Samples

![](images/normalized1.png)

## Performance

From the dataset we got the histogram of the images using our class [ProcessNormalized](https://github.com/suayder/computer_vision_eps_usp/blob/d902bf9fddb770d24f6c38995e9cebda4b7f0666/src/normalize.py#L60) which processed the normalized (or not normalized) images and plot the mean of the values and the variances. To do this we used numpy and skimage libraries to process each value.

In the code, basicaly we go through each class of image in the method [process_by_class](https://github.com/suayder/computer_vision_eps_usp/blob/d902bf9fddb770d24f6c38995e9cebda4b7f0666/src/normalize.py#L87) and called the methods [build_mean_prototype](https://github.com/suayder/computer_vision_eps_usp/blob/d902bf9fddb770d24f6c38995e9cebda4b7f0666/src/normalize.py#L73) and [histogram_stats_by_class](https://github.com/suayder/computer_vision_eps_usp/blob/d902bf9fddb770d24f6c38995e9cebda4b7f0666/src/normalize.py#L151) that do the computation respectively of the mean prototype of the image (which is the mean of all images of each class) and mean and variance of the histogram in the class.
