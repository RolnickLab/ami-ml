## Model Prediction on Trap Data: Localization, Classification, and Tracking

This folder contains scripts for running model prediction on the raw trap data. There are two options:

### Option 1: Sequential 
The following steps need to be run in a sequence:

1. `localization_classification.py`: Localization and classification on the raw images.

```bash
python localization_classification.py \
--data_dir '/home/mila/a/aditya.jain/scratch/TrapData_QuebecVermont_2022/Quebec/' \
--image_folder '2022_05_18' \
--image_resize 300 \
--model_localize_type fasterrcnn_resnet50_fpn \
--model_localize 'v1_localization_model_fasterrcnn_resnet50_fpn_2021-08-17-12-06.pt' \
--localize_score_thresh 90 \
--model_moth '/home/mila/a/aditya.jain/logs/quebec-vermont-moth-model_v02_resnet50_2022-08-01-07-33.pt' \
--model_moth_type 'resnet50' \
--model_moth_nonmoth '/home/mila/a/aditya.jain/logs/moth-nonmoth-effv2b3_20220506_061527_30.pth' \
--model_moth_nonmoth_type 'tf_efficientnetv2_b3' \
--category_map_moth '/home/mila/a/aditya.jain/logs/quebec-vermont-moth_category-map_4Aug2022.json' \
--category_map_moth_nonmoth '/home/mila/a/aditya.jain/logs/05-moth-nonmoth_category_map.json'
```
The description of arguments:
* `--data_dir`: path to the root directory containing the trap data. **Required**.
* `--image_folder`: date folder within root directory containing the images. **Required**.
* `--image_resize`: image resize for classification. **Optional**. Default is **224**.
* `--model_localize_type`: pytorch object detector type. **Required**. (See below for details)
* `--model_localize`: path to the localization model. **Required**.
* `--localize_score_thresh`: confidence threshold of the object detector over which to consider predictions. **Optional**. Default is 99.
* `--model_moth`: path to the fine-grained moth classification model. **Required**.
* `--model_moth_type`: type of model used; currently accepts resnet50 or tf_efficientnetv2_b3. **Required**.
* `--model_moth_nonmoth`: path to the moth-nonmoth model. **Required**.
* `--model_moth_nonmoth_type`: type of model used; currently accepts resnet50 or tf_efficientnetv2_b3. **Required**.
* `--category_map_moth`: path to the moth category map for converting integer labels to name labels. **Required**.
* `--category_map_moth_nonmoth`: path to the moth-nonmoth category map for converting integer labels to name labels. **Required**.

Currently, four types of object localization models are trained and `--model_localize_type` needs to be passed accordingly:
* `fasterrcnn_resnet50_fpn`: Faster R-CNN model with a ResNet-50-FPN backbone [(link)](https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html#torchvision.models.detection.fasterrcnn_resnet50_fpn)
* `ssdlite320_mobilenet_v3_large`: SSDlite model architecture with input size 320x320 and a MobileNetV3 Large backbone [(link)](https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html#torchvision.models.detection.ssdlite320_mobilenet_v3_large)
* `fasterrcnn_mobilenet_v3_large_320_fpn`: Low resolution Faster R-CNN model with a MobileNetV3-Large backbone tuned for mobile use cases [(link)](https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn.html#torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn)
* `fasterrcnn_mobilenet_v3_large_fpn`: Faster R-CNN model with a MobileNetV3-Large FPN backbone [(link)](https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn.html#torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn)

`fasterrcnn_resnet50_fpn` has the best accuracy currently and its use is strongly encouraged amongst all.

<br>

2. `tracks_w_classification_multiple.py`: Given localization and classification annotation for the image sequence, perform tracking using CNN features, IoU, distance and box ratio.

```bash
python tracks_w_classification_multiple.py \
--data_dir '/home/mila/a/aditya.jain/scratch/TrapData_QuebecVermont_2022/Quebec/' \
--image_folder '2022_05_18' \
--model_moth_cnn '/home/mila/a/aditya.jain/logs/quebec-vermont-moth-model_v02_resnet50_2022-08-01-07-33.pt' \
--category_map_moth '/home/mila/a/aditya.jain/logs/quebec-vermont-moth_category-map_4Aug2022.json' \
--image_resize 300 \
--weight_cnn 1 \
--weight_iou 1 \
--weight_box_ratio 1 \
--weight_distance 1 
```

The description of arguments:
* `--data_dir`: path to the root directory containing the trap data. **Required**.
* `--image_folder`: date folder within the root directory containing the images. **Required**.
* `--model_moth_cnn`: path to the moth model for comparison of cnn features. **Required**.
* `--category_map_moth`: path to the moth category map for converting integer labels to name labels. **Required**.
* `--image_resize`: image resize for classification. **Optional**. Default is **224**.
* `--weight_cnn`: weight factor on the cnn features. **Optional**. Default is **1**.
* `--weight_iou`: weight factor on the intersection over union. **Optional**. Default is **1**.
* `--weight_box_ratio`: weight factor on the ratio of the box areas. **Optional**. Default is **1**.
* `--weight_distance`: weight factor on the distance between boxes. **Optional**. Default is **1**.

<br>

3. `video.py`: Make a video using the localization, classification, and tracking information.

```bash
python video.py \
--data_dir '/home/mila/a/aditya.jain/scratch/TrapData_QuebecVermont_2022/Quebec/' \
--image_folder '2022_05_18' \
--frame_rate 5 \
--scale_factor 0.5 \
--region 'Quebec'
```
The description of  arguments:
* `--data_dir`: path to the root directory containing the trap data. **Required**.
* `--image_folder`: date folder within root directory containing the images. **Required**.
* `--frame_rate`: frame rate of the resulting video. **Optional**. Default is **5**.
* `--scale_factor`: scale the raw image by this factor before stitching into the video. **Optional**. Default is **0.5**.
* `--region`: name of the region where the data came from (to be used for output video name). **Required**.



### Option 2: End-to-End 
The other option is to run `end_to_end.py` which does everything in a single script.

```bash
python end_to_end.py \
--data_dir '/home/mila/a/aditya.jain/scratch/TrapData_QuebecVermont_2022/Quebec/' \
--image_folder '2022_05_18' \
--model_localize_type fasterrcnn_resnet50_fpn \
--model_localize 'v1_localization_model_fasterrcnn_resnet50_fpn_2021-08-17-12-06.pt' \
--localize_score_thresh 90 \
--model_moth '/home/mila/a/aditya.jain/logs/quebec-vermont-moth-model_v02_resnet50_2022-08-01-07-33.pt' \
--model_moth_type 'resnet50' \
--model_moth_nonmoth '/home/mila/a/aditya.jain/logs/moth-nonmoth-effv2b3_20220506_061527_30.pth' \
--model_moth_nonmoth_type 'tf_efficientnetv2_b3' \
--category_map_moth '/home/mila/a/aditya.jain/logs/quebec-vermont-moth_category-map_4Aug2022.json' \
--category_map_moth_nonmoth '/home/mila/a/aditya.jain/logs/05-moth-nonmoth_category_map.json' \
--model_moth_cnn '/home/mila/a/aditya.jain/logs/v01_mothmodel_2021-06-08-04-53.pt' \
--image_resize 300 \
--weight_cnn 1 \
--weight_iou 1 \
--weight_box_ratio 1 \
--weight_distance 1 \
--frame_rate 5 \
--scale_factor 0.5 \
--region 'Quebec'
```

The description of arguments:
* `--data_dir`: path to the root directory containing the trap data. **Required**.
* `--image_folder`: date folder within root directory containing the images. **Required**.
* `--model_localize_type`: pytorch object detector type. **Required**. 
* `--model_localize`: path to the localization model. **Required**.
* `--localize_score_thresh`: confidence threshold of the object detector over which to consider predictions. **Optional**. Default is 99.
* `--model_moth`: path to the fine-grained moth classification model. **Required**.
* `--model_moth_type`: type of model used; currently accepts resnet50 or tf_efficientnetv2_b3. **Required**.
* `--model_moth_nonmoth`: path to the moth-nonmoth model. **Required**.
* `--model_moth_nonmoth_type`: type of model used; currently accepts resnet50 or tf_efficientnetv2_b3. **Required**.
* `--category_map_moth`: path to the moth category map for converting integer labels to name labels. **Required**.
* `--category_map_moth_nonmoth`: path to the moth-nonmoth category map for converting integer labels to name labels. **Required**.
* `--model_moth_cnn`: path to the moth model for comparison of cnn features. **Required**.
* `--image_resize`: image resize for classification. **Optional**. Default is **224**.
* `--weight_cnn`: weight factor on the cnn features. **Optional**. Default is **1**.
* `--weight_iou`: weight factor on the intersection over union. **Optional**. Default is **1**.
* `--weight_box_ratio`: weight factor on the ratio of the box areas. **Optional**. Default is **1**.
* `--weight_distance`: weight factor on the distance between boxes. **Optional**. Default is **1**.
* `--frame_rate`: frame rate of the resulting video. **Optional**. Default is **5**.
* `--scale_factor`: Scale the raw image by this factor before stitcing into the video. **Optional**. Default is **0.5**.
* `--region`: name of the region where the data came from (to be used for output video name). **Required**.