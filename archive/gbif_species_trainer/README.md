# Species model trainer from GBIF 

This repository contains the code to download image and metadata for a list of species from [Global Biodiversity Information Facility](https://www.gbif.org/) (GBIF) and train a deep learning model using the downloaded data.

## Setup python environment
Install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) and prepare a python environment using the following steps:

Build a new conda environment
```bash
conda create -n milamoth_ai python=3.9
```

Activate the conda environment:
```bash
conda activate milamoth_ai
```

Install cuda toolkit and pytorch
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

Install additional libraries using pip:

```bash
python3 -m pip install -r requirements.txt
```

## Download Data
Follow instructions in the `data_download` folder to download data for a list of species.

## Train Model
Train a model using scripts in `model_training` folder.
