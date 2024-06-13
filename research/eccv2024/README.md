# ECCV 2024 AMI-Traps related Code
Export of the AMI-Traps dataset to standard formats and model evaluation for the ECCV 2024 conference.

## Table of Contents
| Script or Folder | Function |
| -------- | -------- |
| `model_evaluation/`  | Evaluation of AMI-GBIF trained models on the AMI-Traps dataset  |
| `analyze_data.py`  | Analyze various taxonomy related annotation statistics for the AMI-Traps dataset   |
| `export_to_webdataset_and_crops.py`  | Export the dataset into the webdataset format and also raw individual crops along with a labels file   |
| `export_to_yolo.py`  | Export the dataset into the standard YOLO format   |
| `process_label_file.py`  | Add the rank and hierarchy information to the label file in the YOLO-format dataset. Doesn't require to be re-run  |
| `test_download_raw_images.py`  | Test the downloading of raw camera trap images from the annotation file  |
| `test_webdataset.py`  | Test the retrieval of the insect crops and labels from the webdataset files  |
| `test_yolo_annotations.py`  | Test the bounding box and label annotations stored in the YOLO format |

Note:
- The annotations file is private and comes in a JSON format. Please contact the repository owner for access.
- The python scripts can be run through the command line or submitted to a GPU cluster using a bash script. Some example bash scripts are provided.
