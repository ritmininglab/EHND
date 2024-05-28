# Evidential Hierarchical Novelty Detection
It provides the implementation of Evidential Hierarchical Novelty Detection(E-HND). 

## Requirements
---------------------

* Python
* Conda

Install pytorch using the [Official Website](https://pytorch.org/get-started/locally/). Install the additional requirements:
* h5py
* scipy

## Datasets
---------------------
The known-novel split are provided in the respective files as:

Known Classes:
* CUB : `./taxonomy/CUB/trainvalclasses.txt`
* AWA2 : `./taxonomy/AWA/trainvalclasses.txt`
* Tiny Imagenet : `./taxonomy/ImageNet/classes_known.txt`

Novel Classes:
* CUB : `./taxonomy/CUB/testclasses.txt`
* AWA2 : `./taxonomy/AWA/testclasses.txt`
* Tiny Imagenet : `./taxonomy/ImageNet/classes_novel.txt`

The taxonomy for each dataset can be created by the command:

``` {.sourceCode .text}
sh scripts/preparation.sh <dataset> ehnd
```
where, `<dataset>` is `cub`, `awa2` or `imagenet`.

Create the `datasets` folder. For each datasets, create a separate folder. Then, extract resnet101 features of the datasets for splits as:

* `resnet101_train.h5` : Training images from known classes.
* `resnet101_val.h5`: Validation images from known classes.
* `resnet101_known.h5` : Test images from known classes.
* `resnet101_novel.h5` : Test images from novel classes.


The original datasets can be download from the following link:
* CUB : [Official Website](https://www.vision.caltech.edu/datasets/cub_200_2011/)
* AWA2 : [Official Website](https://cvml.ista.ac.at/AwA2/)
* Tiny Imagenet : [Download Link](http://cs231n.stanford.edu/tiny-imagenet-200.zip)

## Running the experiments
---------------------
For CUB dataset, E-HND

``` {.sourceCode .text}
sh scripts/train.sh cub ehnd
```

For CUB dataset, TD+E-HND

``` {.sourceCode .text}
sh scripts/train.sh cub td
sh scripts/train.sh cub td+ehnd
```

For AWA2 dataset, E-HND
``` {.sourceCode .text}
sh scripts/train.sh awa2 ehnd
```

For AWA2 dataset, TD+E-HND
``` {.sourceCode .text}
sh scripts/train.sh awa2 td
sh scripts/train.sh awa2 td+ehnd
```

For Tiny Imagenet dataset, E-HND
``` {.sourceCode .text}
sh scripts/train.sh imagenet ehnd
```

For Tiny Imagenet dataset, TD+E-HND
``` {.sourceCode .text}
sh scripts/train.sh imagenet td
sh scripts/train.sh imagenet td+ehnd
```