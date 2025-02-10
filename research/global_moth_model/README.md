# Global Moth Model
Research related to the development of a global moth species classification model for automated moth monitoring.

## Process
The below steps are carrried out to train a global model.  

### Checklist preparation
1. **Fetch Leps Checklist**: Download the Lepidoptera taxonomy from GBIF ([DOI](https://doi.org/10.15468/dl.jzy3de)).
2. **Fetch DwC-A**: Fetch the Darwin Core Archive from GBIF for the order Lepidoptera ([DOI](https://doi.org/10.15468/dl.6j5bzj)). 
3. **Curate Moth Checklist** (`prepare_gbif_checklist.py`): Clean and curate the Lepidoptera checklist to have only moth species. Remove all non-species taxa and butterfly families. A curated list is [here](https://docs.google.com/spreadsheets/d/1E6Zn2hXbHGMMAiPhtDXFO9_hDtl68lG5fx2vg0jyBvg/edit?usp=sharing).

### Dataset download and curation
The next steps to download and curate data are followed from [here](https://github.com/RolnickLab/ami-ml/tree/main/src/dataset_tools).

1. **Fetch GBIF images**: Download the images from GBIF using the command `ami-dataset fetch-images`. An example slurm script with the argument options is provided (`job_fetch_images.sh`). The DwC-A file requires about 300GB of RAM to be loaded. There should be smarter ways to load the archive file in (multiple?) smaller memory but we haven't explored it ourselves.
2. **Verify images**: Verify the downloaded images for corruption (`job_verify_images.sh`).
3. **Delete corrupted images**: `job_delete_images.sh`
4. **Lifestage prediction:** Run the lifestage prediction model on images without the lifestage tag. The purpose is to remove non-adult moth images from the dataset (`job_predict_lifestage.sh`).
5. **Final clean dataset:** Create the final list of images cleaned after image verification and lifestage prediction (`job_clean_dataset.sh`).
6. **Dataset splits:** Create dataset splits for model training (`job_split_dataset.sh`).

### Model training
A non-public code was used to train this model but a refurbished version is now available [here](https://github.com/RolnickLab/ami-ml/tree/main/src/classification).

