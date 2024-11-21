# Insect Species Classification
Scripts for training vision classification models for species identification in insect camera traps.

## Overview
Although this code is primarily developed for training insect classification models, it is designed to be highly generic. Therefore, it can be easily adapted or modified to train any general deep learning model.



## Usage
Run the following command from anywhere within the root `ami-ml` package to start the training script:
```bash
ami-classification train-model [OPTIONS]
```

For more information on the available options:
```bash
ami-classification train-model --help
```

The training workflow assumes the dataset to be in [WebDataset](https://webdataset.github.io/webdataset/) format. WebDataset makes loading of large datasets efficient and faster. For the use-case of insect classification models, assembling of the dataset in WebDataset format is discussed in the [`dataset-tools`](https://github.com/RolnickLab/ami-ml/tree/main/src/dataset_tools) package. 

*Optional use*: Learning rate scheduler, mixed resolution data augmentation technique, and Weights & Biases (W&B) for experiment management.


## Installation
The python environment setup is described in the [Setup guide](https://github.com/RolnickLab/ami-ml/tree/main?tab=readme-ov-file#setup) on the main repository page. The use of Conda + Poetry is recommended.


## To-Do for Next Update
- Calculate macro-averaged accuracy (average of individual class accuracy)
- Upload species checklist with individual class accuracy on W&B
- An option to train with standard dataset formats i.e. not in WebDataset format  


## How to Contribute?
For any bugs or feature suggestions, please raise an issue. Any substantial improvements are welcome through a pull request!