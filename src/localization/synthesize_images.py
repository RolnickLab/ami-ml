"""


"""

import json
import os
import re
import typing as tp
from itertools import cycle, islice
from operator import itemgetter

import click
import numpy as np
from data.custom_datasets import DatasetAsNumpyArrays
from PIL import Image
from tqdm import tqdm


@click.command(context_settings={"show_default": True})
@click.option("--nb_new_images_per_background", type=int, default=2)
@click.option("--nb_crops_per_image", type=int, default=20)
@click.option("--random_seed", type=int, default=42)
@click.option("--split_ratio", type=float, default=None)
@click.option("--save_bboxes", type=bool, default=True)
@click.option("--x_padding", type=int, default=0)
@click.option("--y_padding", type=int, default=0)
@click.option(
    "--crops_path",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
)
@click.option(
    "--backgrounds_path",
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    "--save_dir",
    type=click.Path(exists=True, file_okay=False),
    required=True,
)
def main(
    nb_new_images_per_background: int,
    nb_crops_per_image: int,
    random_seed: int,
    split_ratio: tp.Optional[float],
    save_bboxes: bool,
    x_padding: int,
    y_padding: int,
    crops_path: str,
    backgrounds_path: str,
    save_dir: str,
):
    rng = np.random.default_rng(random_seed)
    background_dataset = DatasetAsNumpyArrays(backgrounds_path)
    crops, masks = load_crops_and_masks(crops_path, rng=rng)
    idxs_at_iter = create_lists_of_indices(
        nb_new_images_per_background * len(background_dataset),
        nb_crops_per_image,
        len(crops),
    )
    nb_crops = len(crops)
    bboxes = {}

    iter = 0
    for background, background_name in tqdm(background_dataset):
        for i in range(nb_new_images_per_background):
            # select moths
            item_getter = itemgetter(*idxs_at_iter[iter])
            selected_crops = item_getter(crops)
            selected_masks = item_getter(masks)

            rot90_k, flip_axis = select_transform(iter, nb_crops_per_image, nb_crops)
            # synthesize new image
            new_image, image_bboxes = synthesize_image(
                background,
                selected_crops,
                selected_masks,
                rng,
                save_bboxes,
                rot90_k=rot90_k,
                flip_axis=flip_axis,
                padding=(x_padding, y_padding),
            )

            # save image in save_dir
            image_name = os.path.splitext(background_name)[0] + f"_synth_{i}.jpg"
            Image.fromarray(new_image).save(os.path.join(save_dir, image_name))

            if save_bboxes:
                bboxes[image_name] = image_bboxes

            iter = iter + 1

    if save_bboxes:
        with open(os.path.join(save_dir, "synthesized_data_gt.json"), "w") as f:
            json.dump(bboxes, f)
    return


def select_transform(
    iter: int, nb_crops_per_image: int, nb_crops: int
) -> tp.Tuple[int, tp.Optional[int]]:
    """Instead of applying a random rotation or flip for each crop, we first use all
    crops without transformation, then we use all crops again with a 90° rotation,
    then with a 180° rotation... And so on. This way, diversity is maximized. This
    function is used to select the proper transform at a given iteration."""

    current_cyle = (iter * nb_crops_per_image // nb_crops) % 6
    if current_cyle == 0:
        rot90_k = 0
        flip_axis = None
    elif current_cyle == 1:
        rot90_k = 1
        flip_axis = None
    elif current_cyle == 2:
        rot90_k = 2
        flip_axis = None
    elif current_cyle == 3:
        rot90_k = 3
        flip_axis = None
    elif current_cyle == 4:
        rot90_k = 0
        flip_axis = 0
    elif current_cyle == 5:
        rot90_k = 0
        flip_axis = 1
    return rot90_k, flip_axis


def synthesize_image(
    background_image: np.ndarray,
    crops: tp.List[np.ndarray],
    masks: tp.List[np.ndarray],
    rng: np.random.Generator,
    save_bboxes: bool,
    rot90_k: int = 0,
    flip_axis: tp.Optional[int] = None,
    padding: tp.Tuple[int, int] = (0, 0),
) -> tp.Tuple[np.ndarray, tp.Optional[tp.List[tp.Any]]]:
    if len(crops) != len(masks):
        raise ValueError("Same number of crops and masks is required")

    heigth, width = background_image.shape[0:2]
    new_image = background_image.copy()
    crops_coordinates = []  # Coordinates of processed crops in the image
    bboxes = [[], []]

    for crop, mask in zip(crops, masks):
        if crop.shape[0:2] != mask.shape:
            raise ValueError("Corresponding crops and masks must have the same shape")

        # Apply transforms
        crop = np.rot90(crop, k=rot90_k)
        mask = np.rot90(mask, k=rot90_k)
        if flip_axis is not None:
            crop = np.flip(crop, axis=flip_axis)
            mask = np.flip(mask, axis=flip_axis)

        overlap = True
        while overlap:
            # Generate 'landing' coordinates for crop
            x = rng.integers(heigth) - crop.shape[0] // 2
            y = rng.integers(width) - crop.shape[1] // 2

            if len(crops_coordinates) == 0:
                overlap = False
            else:  # Check if there are overlaps
                for i, (x_i, y_i) in enumerate(crops_coordinates):
                    overlap = check_overlap(mask, x, y, masks[i], x_i, y_i)
                    if overlap:
                        break

        crops_coordinates.append((x, y))

        # Paste the crop on the image
        paste_crop(new_image, crop, mask, x, y)

        if save_bboxes:
            h, w = crop.shape[0:2]
            x2 = int(min(x + h + padding[1] / 2, heigth))
            y2 = int(min(y + w + padding[0] / 2, width))
            x1 = int(max(0, x - padding[1] / 2))
            y1 = int(max(0, y - padding[0] / 2))
            bboxes[0].append([y1, x1, y2, x2])
            bboxes[1].append(1)

    return new_image, bboxes


def paste_crop(image: np.ndarray, crop: np.ndarray, mask: np.ndarray, x: int, y: int):
    indices1, indices2 = find_overlap(image.shape, 0, 0, mask.shape, x, y)
    start_x1, end_x1, start_y1, end_y1 = indices1
    start_x2, end_x2, start_y2, end_y2 = indices2

    try:
        image[start_x1:end_x1, start_y1:end_y1][
            mask[start_x2:end_x2, start_y2:end_y2]
        ] = crop[start_x2:end_x2, start_y2:end_y2][
            mask[start_x2:end_x2, start_y2:end_y2]
        ]
    except IndexError:
        breakpoint()
    return


def check_overlap(
    mask1: np.ndarray,
    x1: int,
    y1: int,
    mask2: np.ndarray,
    x2: int,
    y2: int,
    overlap_thresh: float = 0.15,
) -> bool:
    try:
        indices1, indices2 = find_overlap(mask1.shape, x1, y1, mask2.shape, x2, y2)
    except ValueError:
        return False

    start_x1, end_x1, start_y1, end_y1 = indices1
    start_x2, end_x2, start_y2, end_y2 = indices2

    area_overlap = np.sum(
        np.logical_and(
            mask1[start_x1:end_x1, start_y1:end_y1],
            mask2[start_x2:end_x2, start_y2:end_y2],
        )
    )
    area1 = np.sum(mask1)
    area2 = np.sum(mask2)

    if area_overlap > overlap_thresh * min(area1, area2):
        return True
    else:
        return False


def find_overlap(
    shape1: tp.Tuple[int, int],
    x1: int,
    y1: int,
    shape2: tp.Tuple[int, int],
    x2: int,
    y2: int,
):
    h1, w1 = shape1[0:2]
    h2, w2 = shape2[0:2]

    if (x2 - x1 > h1) or (x1 - x2 > h2) or (y2 - y1 > w1) or (y1 - y2 > w2):
        raise ValueError("No overlap")

    if x2 - x1 > 0:  # Box 1 is above box 2
        start_x1 = x2 - x1
        end_x1 = min(h1, h2 - (x1 - x2))
        start_x2 = 0
        end_x2 = min(h2, h1 - (x2 - x1))
    else:  # Box 1 is below box 2
        start_x1 = 0
        end_x1 = min(h1, h2 - (x1 - x2))
        start_x2 = x1 - x2
        end_x2 = min(h2, h1 + (x1 - x2))

    if y2 - y1 > 0:  # Box 1 is left of box 2
        start_y1 = y2 - y1
        end_y1 = min(w1, w2 - (y1 - y2))
        start_y2 = 0
        end_y2 = min(w2, w1 - (y2 - y1))
    else:  # Box 1 is right of box 2
        start_y1 = 0
        end_y1 = min(w1, w2 - (y1 - y2))
        start_y2 = y1 - y2
        end_y2 = min(w2, w1 + (y1 - y2))

    return (start_x1, end_x1, start_y1, end_y1), (start_x2, end_x2, start_y2, end_y2)


def load_crops_and_masks(
    crops_path: str, rng: tp.Optional[np.random.Generator] = None
) -> tp.Tuple[tp.List[np.ndarray], tp.List[np.ndarray]]:
    """Returns the crops and their corresponing masks given the path to the npz file.
    The npz file is expected to store (crop, mask) pairs under key pairs with the format
    ('xxx_crop_yyy', 'xxx_mask_yyy'), where 'xxx' and 'yyy' can be anything as long as
    they are the same in a pair. The order of the dictionary is not critical."""

    data = np.load(crops_path)
    keys = list(data.keys())
    crop_keys = sorted([s for s in keys if re.search(r"_crop_", s)])
    mask_keys = sorted([s for s in keys if re.search(r"_mask_", s)])

    if rng is not None:
        permutation = rng.permutation(len(crop_keys))
        crop_keys = list(np.array(crop_keys)[permutation])
        mask_keys = list(np.array(mask_keys)[permutation])

    crops, masks = [], []
    for crop_key, mask_key in zip(crop_keys, mask_keys):
        crops.append(data[crop_key])
        masks.append(data[mask_key])

    return crops, masks


def create_lists_of_indices(
    nb_iters: int, elements_per_step: int, original_list_length: int
) -> tp.List[tp.List]:
    """Returns list of {nb_iters} lists, each with {elements_per_step} indices. These
    can be used to access an original list of length {original_list_lenght}"""
    final_list = []
    indices = cycle(range(original_list_length))

    for _ in range(nb_iters):
        indices_at_iter = list(islice(indices, elements_per_step))
        final_list.append(indices_at_iter)
    return final_list


if __name__ == "__main__":
    main()
