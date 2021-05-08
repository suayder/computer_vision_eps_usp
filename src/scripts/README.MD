This folder contains some utils scripts used to prepare the dataset, is very specific to our dataset organization.

in summary:

- `make_dirs.ipynp` is just to build a tree of directories to organize the images

- `stage_to_dataset.ipynp` this will do the most important thing, read the images from these folders, copy them to a dataset folder simplified and generate the metadata of the images, then the dataset is ready.