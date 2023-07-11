"""This script is used to run the Segment Anything Model (SAM) on a dataset. It can
produce two outpus:
- bounding boxes, in the same format as the object detection models of the
inference_localization.py script
- crops of moths, with their mask.

Usage:
    python3 inference_segmentation.py [OPTIONS]

Options:
    --ckpt_path (str)
    --data_dir (str): where the jpg images are located.
    --save_dir (str): where to save the outuput. By default, this is equal to data_dir
    --model_type (str): by default, vit_l. Needs to match the model given at the
        checkpoint path. From smallest to largest, three options are possible: vit_b,
        vit_l, and vit_h. According to the paper, vit_h improves substantially over
        vit_b, but has only marginal gains over vit_l.
    --sampling_rate (int): by default, 1. Can set a higher value if you don't want to
        run inferences on every image, but just on one in {sampling_rate}
    --predict_bboxes (bool): whether to use SAM as an object detector. In this case, the
        script outputs a json file in the same format as inference_localization.py does.
        The json file is saved in save_dir. By default, True.
    --generate_crops (bool): whether to generate crops. These are stored in save_dir as
        a .npz file, with keywords <image ID>_crop_<ID> and <image ID>_mask_<ID> for
        crops and masks, respectively. A crop is an image saved as a (H x W x C) uint8
        array, while a mask is saved as a (H x W) bool array.
    --save_image_list (bool): whether to save the names of all processed images at a txt
        file. This is provided to help the user. The image names can also be recovers
        from the generated crops or bboxes. By default, True.
    --stats (bool): by default, True.
    --[...] SamAutomaticGenerator parameters. Check out their GitHub repo at:
    https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/automatic_mask_generator.py
"""

import json
import os
import typing as tp

import click
import numpy as np
import torch
from tqdm import tqdm
from data.custom_datasets import DatasetAsNumpyArrays
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from utils import SAMtypes, compute_model_size


@click.command(context_settings={"show_default": True})
@click.option(
    "--model_type",
    type=click.Choice(tp.get_args(SAMtypes)),
    default="vit_l",
)
@click.option("--sampling_rate", type=int, default=1)
@click.option(
    "--ckpt_path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
)
@click.option(
    "--data_dir",
    type=click.Path(exists=True, file_okay=False),
    required=True,
)
@click.option(
    "--save_dir",
    type=click.Path(exists=True, file_okay=False),
    default=None,
)
@click.option("--predict_bboxes", type=bool, default=True)
@click.option("--generate_crops", type=bool, default=True)
@click.option("--save_image_list", type=bool, default=True)
@click.option("--stats", type=bool, default=True)
@click.option("--points_per_side", type=int, default=32)
@click.option("--points_per_batch", type=int, default=64)
@click.option("--pred_iou_thresh", type=float, default=0.88)
@click.option("--stability_score_thresh", type=float, default=0.95)
@click.option("--stability_score_offset", type=float, default=1.0)
@click.option("--box_nms_thresh", type=float, default=0.7)
@click.option("--crop_n_layers", type=int, default=0)
@click.option("--crop_nms_thresh", type=float, default=0.7)
@click.option("--crop_overlap_ratio", type=float, default=512 / 1500)
@click.option("--crop_n_points_downscale_factor", type=int, default=1)
@click.option("--min_mask_region_area", type=int, default=0)
@click.option(
    "--output_mode",
    type=click.Choice(["binary_mask", "uncompressed_rle", "coco_rle"]),
    default="binary_mask",
)
def main(
    data_dir: str,
    save_dir: tp.Optional[str],
    ckpt_path: tp.Optional[str],
    model_type: SAMtypes,
    sampling_rate: int,
    predict_bboxes: bool,
    generate_crops: bool,
    save_image_list: bool,
    stats: bool,
    points_per_side: tp.Optional[int],
    points_per_batch: int,
    pred_iou_thresh: float,
    stability_score_thresh: float,
    stability_score_offset: float,
    box_nms_thresh: float,
    crop_n_layers: int,
    crop_nms_thresh: float,
    crop_overlap_ratio: float,
    crop_n_points_downscale_factor: int,
    min_mask_region_area: int,
    output_mode: str,
    point_grids: tp.Optional[tp.List[np.ndarray]] = None,
):
    if save_dir == None:
        save_dir = data_dir
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        raise Exception("GPU is needed to run SAM model")

    if generate_crops is True and output_mode != "binary_mask":
        raise Exception("Please use 'binary_mask' output mode when generating crops")

    sam = sam_model_registry[model_type](checkpoint=ckpt_path)
    sam.to(device=device)

    if stats:
        size = compute_model_size(sam)
        print(f"Size of loaded {model_type} is: {size:.2f}MB")

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=points_per_side,
        points_per_batch=points_per_batch,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        stability_score_offset=stability_score_offset,
        box_nms_thresh=box_nms_thresh,
        crop_n_layers=crop_n_layers,
        crop_nms_thresh=crop_nms_thresh,
        crop_overlap_ratio=crop_overlap_ratio,
        crop_n_points_downscale_factor=crop_n_points_downscale_factor,
        point_grids=point_grids,
        min_mask_region_area=min_mask_region_area,
        output_mode=output_mode,
    )
    dataset = DatasetAsNumpyArrays(data_dir, sampling_rate=sampling_rate)

    preds = {}
    crops = {}
    image_names = []
    for image, image_name in tqdm(dataset):
        output = mask_generator.generate(image)

        if save_image_list:
            image_names.append(image_name)

        if predict_bboxes:
            bboxes_data = sam_output_to_bboxes(
                output, image, area_ratio_upper_thr=0.25, area_ratio_lower_thr=0.0002
            )  # 0.0002 corresponds to 40x40 square with a 4096x2160 image
            preds[image_name] = bboxes_data

        if generate_crops:
            crops_i = sam_output_to_crops(
                output,
                image,
                image_name,
                area_ratio_upper_thr=0.25,
                area_ratio_lower_thr=0.0002,
            )
            crops |= crops_i

    if predict_bboxes:
        with open(
            os.path.join(save_dir, "predictions_" + model_type + ".json"), "w"
        ) as f:
            json.dump(preds, f)

    if generate_crops:
        np.savez(os.path.join(save_dir, "crops_" + model_type), **crops)

    if save_image_list:
        with open(
            os.path.join(save_dir, model_type + "_processed_images.txt"), "w"
        ) as f:
            for name in image_names:
                f.write(name + ",\n")

    return


def sam_output_to_crops(
    sam_output: tp.List[tp.Dict[str, tp.Any]],
    image: np.ndarray,
    image_name: str,
    area_ratio_upper_thr: float = 1,
    area_ratio_lower_thr: float = 0,
):
    crops = {}
    image_name = os.path.splitext(image_name)[0]  # Remove '.jpg' extension
    area_upper_thr = image.shape[0] * image.shape[1] * area_ratio_upper_thr
    area_lower_thr = image.shape[0] * image.shape[1] * area_ratio_lower_thr

    for i, mask in enumerate(sam_output):
        if mask["area"] > area_upper_thr or mask["area"] < area_lower_thr:
            continue
        # x-axis is the horizontal axis in the image, i.e. the second dimension in the
        # numpy array
        x_min, y_min, x_max, y_max = xywh_to_xyxy(mask["bbox"])
        crops[f"{image_name}_crop_{i}"] = image[y_min : y_max + 1, x_min : x_max + 1]
        crops[f"{image_name}_mask_{i}"] = mask["segmentation"][
            y_min : y_max + 1, x_min : x_max + 1
        ]
    return crops


def sam_output_to_bboxes(
    sam_output: tp.List[tp.Dict[str, tp.Any]],
    image: np.ndarray,
    area_ratio_upper_thr: float = 1,
    area_ratio_lower_thr: float = 0,
):
    bboxes = []
    scores = []
    labels = []
    area_upper_thr = image.shape[0] * image.shape[1] * area_ratio_upper_thr
    area_lower_thr = image.shape[0] * image.shape[1] * area_ratio_lower_thr

    for mask in sam_output:
        if mask["area"] > area_upper_thr or mask["area"] < area_lower_thr:
            continue
        bbox = mask["bbox"]
        bboxes.append(xywh_to_xyxy(bbox))
        score = geometric_mean(
            mask["predicted_iou"], mask["stability_score"], clip=True
        )
        scores.append(score)
        labels.append(1)

    return [bboxes, labels, scores]


def xywh_to_xyxy(box):
    new_box = box.copy()
    new_box[2] = new_box[0] + new_box[2]
    new_box[3] = new_box[1] + new_box[3]
    return new_box


def geometric_mean(x: float, y: float, clip: bool = False) -> float:
    mean = np.sqrt(x * y)
    if clip and mean > 1:
        mean = 1
    return mean


if __name__ == "__main__":
    main()
