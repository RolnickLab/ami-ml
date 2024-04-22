"""
Command Line Interface for the dataset tools module

To add a new command, create a new function below following these instructions:

- Create a new Command key constant
   <NEW_COMMAND_NAME>_CMD = "<NEW_COMMAND_NAME>_cmd"

- Add the new Command key constant to the COMMAND_KEYS frozenset

- Add the command name and help text in the COMMANDS and COMMANDS_HELP dictionaries

- Create a new function named <COMMAND_NAME>_command(), alongside the appropriate
  options
    - Make sure to reuse appropriate options before creating duplicate ones.
    - The contents of the CLI command should be minimal : execution of an imported
      function

- If unsure, take some time to look at how other commands have been implemented
"""
import functools

import click

from src.dataset_tools.clean_dataset import clean_dataset
from src.dataset_tools.create_webdataset import create_webdataset
from src.dataset_tools.delete_images import delete_images
from src.dataset_tools.fetch_images import fetch_images
from src.dataset_tools.predict_lifestage import predict_lifestage
from src.dataset_tools.split_dataset import split_dataset
from src.dataset_tools.verify_images import verify_images

# Command key constants
# Make sure to add them to COMMAND_KEYS frozenset
CLEAN_CMD = "clean_cmd"
FETCH_CMD = "fetch_cmd"
VERIFY_CMD = "verify_cmd"
DELETE_CMD = "delete_cmd"
PREDICT_CMD = "predict_cmd"
SPLIT_CMD = "split_cmd"
WEBDATASET_CMD = "webdataset_cmd"

# This is most useful to automatically test the CLI
COMMAND_KEYS = frozenset(
    [
        FETCH_CMD,
        VERIFY_CMD,
        DELETE_CMD,
        PREDICT_CMD,
        CLEAN_CMD,
        SPLIT_CMD,
        WEBDATASET_CMD,
    ]
)

# Command dictionary
COMMANDS = {
    FETCH_CMD: "fetch-images",
    VERIFY_CMD: "verify-images",
    DELETE_CMD: "delete-images",
    PREDICT_CMD: "predict-lifestage",
    CLEAN_CMD: "clean-dataset",
    SPLIT_CMD: "split-dataset",
    WEBDATASET_CMD: "create-webdataset",
}

# Command help text dictionary
COMMANDS_HELP = {
    FETCH_CMD: "Download images from urls found in input DwC-A file",
    VERIFY_CMD: "Check if there are errors in the images",
    DELETE_CMD: "Delete images based on input list",
    PREDICT_CMD: "Predict lifestage for moths",
    CLEAN_CMD: "Filter out images to ensure quality of training data",
    SPLIT_CMD: "Split the provided dataset into train, validate and test sets",
    WEBDATASET_CMD: "Assemble final training set in webdataset format",
}


#
# Command decorators for shared command options
#
def with_dwca_file(func):
    @click.option(
        "--dwca-file",
        type=str,
        required=True,
        help="Darwin Core Archive file",
    )
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def with_num_workers(func):
    @click.option(
        "--num-workers",
        type=int,
        default=8,
        help="Number of processes to download in images in parallel",
    )
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def with_verified_data_csv(func):
    @click.option(
        "--verified-data-csv",
        type=str,
        required=True,
        help="CSV file containing verified image info",
    )
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def with_dataset_path(func):
    @click.option(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to directory containing dataset images.",
    )
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def with_results_csv(func):
    @click.option(
        "--results-csv",
        type=str,
        required=True,
        help="File to save image verification info",
    )
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def with_random_seed(func):
    @click.option(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducible experiments",
    )
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


# # # # # # #
# Commands  #
# # # # # # #

# The order of declaration of the commands affect the order
# in which they appear in the CLI


#
# Fetch Images Command
#
@click.command(
    name=COMMANDS[FETCH_CMD],
    help=COMMANDS_HELP[FETCH_CMD],
    context_settings={"show_default": True},
)
@click.option(
    "--dataset-path",
    type=str,
    required=True,
    help="Path to folder where images will be stored",
)
@with_dwca_file
@click.option(
    "--cache-path",
    type=str,
    help=(
        "Folder containing cached images. If provided, the script will try copy"
        " images from cache before trying fetch them."
    ),
)
@click.option(
    "--num-images-per-category",
    type=int,
    default=0,
    help=(
        "Number of images to be downloaded for each category. If not provided, all "
        "images will be fetched."
    ),
)
@with_num_workers
@with_random_seed
@click.option(
    "--request-timeout",
    type=int,
    default=30,
    help="Timout for closing urresponsive connections.",
)
@click.option(
    "--subset-list",
    type=str,
    help=(
        "JSON file with the list of keys to be fetched."
        "If provided, only occorrences with these keys will be fetched. Use the option"
        " --subset_key to define the field to be used for filtering."
    ),
)
@click.option(
    "--subset-key",
    type=str,
    default="acceptedTaxonKey",
    help=(
        "DwC-A field to use for filtering occurrences to be fetched. "
        "See --subset_list"
    ),
)
def fetch_images_command(
    dwca_file: str,
    num_workers: int,
    dataset_path: str,
    cache_path: str,
    subset_list: str,
    subset_key: str,
    num_images_per_category: int,
    request_timeout: int,
    random_seed: int,
):
    fetch_images(
        dwca_file=dwca_file,
        num_workers=num_workers,
        dataset_path=dataset_path,
        cache_path=cache_path,
        subset_list=subset_list,
        subset_key=subset_key,
        num_images_per_category=num_images_per_category,
        request_timeout=request_timeout,
        random_seed=random_seed,
    )


#
# Verify Images Command
#
@click.command(
    name=COMMANDS[VERIFY_CMD],
    help=COMMANDS_HELP[VERIFY_CMD],
    context_settings={"show_default": True},
)
@with_dataset_path
@with_dwca_file
@with_results_csv
@with_num_workers
@click.option(
    "--resume-from-ckpt",
    type=str,
    help=(
        "Checkpoint with partial verification results. If provided, the verification "
        "continues from a previous execution, skipping the already verfied images."
    ),
)
@click.option(
    "--save-freq",
    type=int,
    default=10000,
    help="Save partial verification data every n images",
)
@click.option(
    "--subset-list",
    type=str,
    help=(
        "JSON file with the list of keys to be fetched."
        "If provided, only occorrences with these keys will be fetched. Use the option"
        " --subset_key to define the field to be used for filtering."
    ),
)
@click.option(
    "--subset-key",
    type=str,
    default="acceptedTaxonKey",
    help=(
        "DwC-A field to use for filtering occurrences to be fetched. "
        "See --subset_list"
    ),
)
def verify_images_command(
    dwca_file: str,
    resume_from_ckpt: str,
    save_freq: int,
    num_workers: int,
    dataset_path: str,
    results_csv: str,
    subset_list: str,
    subset_key: str,
):
    verify_images(
        dwca_file=dwca_file,
        resume_from_ckpt=resume_from_ckpt,
        save_freq=save_freq,
        num_workers=num_workers,
        dataset_path=dataset_path,
        results_csv=results_csv,
        subset_list=subset_list,
        subset_key=subset_key,
    )


#
# Delete Images Command
#
@click.command(
    name=COMMANDS[DELETE_CMD],
    help=COMMANDS_HELP[DELETE_CMD],
    context_settings={"show_default": True},
)
@click.option(
    "--error-images-csv",
    type=str,
    required=True,
    help="File to save image verification info",
)
@click.option(
    "--base-path",
    type=str,
    help="Root path for the image list",
)
def delete_images_command(error_images_csv: str, base_path: str):
    delete_images(error_images_csv=error_images_csv, base_path=base_path)


#
# Predict Lifestage Command
#
@click.command(
    name=COMMANDS[PREDICT_CMD],
    help=COMMANDS_HELP[PREDICT_CMD],
    context_settings={"show_default": True},
)
@click.option(
    "--category-map-json",
    type=str,
    required=True,
    help="JSON containing the categories id map.",
)
@with_dataset_path
@click.option("--model-path", type=str, required=True, help="Path to model checkpoint")
@click.option("--num-classes", type=int, required=True, help="Number of categories")
@with_results_csv
@with_verified_data_csv
@click.option(
    "--batch-size", type=int, default=32, help="Batch size used during training."
)
@click.option("--input-size", type=int, default=300, help="Input size of the model")
@click.option(
    "--log-frequence", type=int, default=50, help="Log inferecen every n steps"
)
@click.option(
    "--model-name",
    type=click.Choice(["efficientnetv2-b3"]),
    default="efficientnetv2-b3",
    help="Name of the model",
)
@click.option(
    "--predict-nan-life-stage",
    type=bool,
    default=True,
    help=(
        "You can use this parameter to specify whether you want to make predictions"
        " only for images with the 'life_stage' tag as NaN"
    ),
)
@click.option(
    "--preprocessing-mode",
    type=click.Choice(["tf", "torch", "float32"]),
    default="tf",
    help=(
        "Mode for scaling input: tf scales image between -1 and 1;"
        " torch normalizes inputs using ImageNet mean and std"
        " float32 uses image on scale 0-1"
    ),
)
def predict_lifestage_command(
    verified_data_csv: str,
    dataset_path: str,
    input_size: int,
    preprocessing_mode: str,
    predict_nan_life_stage: bool,
    batch_size: int,
    model_name: str,
    num_classes: int,
    model_path: str,
    log_frequence: int,
    category_map_json: str,
    results_csv: str,
):
    predict_lifestage(
        verified_data_csv=verified_data_csv,
        dataset_path=dataset_path,
        input_size=input_size,
        preprocessing_mode=preprocessing_mode,
        predict_nan_life_stage=predict_nan_life_stage,
        batch_size=batch_size,
        model_name=model_name,
        num_classes=num_classes,
        model_path=model_path,
        log_frequence=log_frequence,
        category_map_json=category_map_json,
        results_csv=results_csv,
    )


#
# Clean Dataset Command
#
@click.command(
    name=COMMANDS[CLEAN_CMD],
    help=COMMANDS_HELP[CLEAN_CMD],
    context_settings={"show_default": True},
)
@with_dwca_file
@with_verified_data_csv
@click.option(
    "--ignore-dataset-by-key",
    type=str,
    default=(
        "f3130a8a-4508-42b4-9737-fbda77748438,"
        "4bfac3ea-8763-4f4b-a71a-76a6f5f243d3,"
        "7e380070-f762-11e1-a439-00145eb45e9a"
    ),
    help=(
        "DatasetKeys separeted by comma. Some datasets might be ignored due to the "
        "poor quality of their images."
    ),
)
@click.option(
    "--life-stage-predictions",
    type=str,
    help=(
        "CSV containing life-stage predictions for images. If provided images without"
        " the GBIF life stage will be filtered based on life stage prediction"
    ),
)
@click.option(
    "--thumb-size",
    type=int,
    default=64,
    help="Minimum side size to an image not be considered as thumbnail",
)
@click.option(
    "--remove-duplicate-url",
    type=bool,
    default=True,
    help="Whether occurrences with duplicate URLs should be removed.",
)
@click.option(
    "--remove-non-adults",
    type=bool,
    default=True,
    help="Whether keeping only occurrences with lifeStage identified as Adult or Imago",
)
@click.option(
    "--remove-tumbnails",
    type=bool,
    default=True,
    help=(
        "Whether small images should be removed. Use the option --thumb_size to "
        "determine the minimum side size."
    ),
)
def clean_dataset_command(
    dwca_file: str,
    verified_data_csv: str,
    remove_duplicate_url: bool,
    ignore_dataset_by_key: str,
    remove_tumbnails: bool,
    thumb_size: int,
    remove_non_adults: bool,
    life_stage_predictions: str,
):
    clean_dataset(
        dwca_file=dwca_file,
        verified_data_csv=verified_data_csv,
        remove_duplicate_url=remove_duplicate_url,
        ignore_dataset_by_key=ignore_dataset_by_key,
        remove_tumbnails=remove_tumbnails,
        thumb_size=thumb_size,
        remove_non_adults=remove_non_adults,
        life_stage_predictions=life_stage_predictions,
    )


#
# Split Dataset Command
#
@click.command(
    name=COMMANDS[SPLIT_CMD],
    help=COMMANDS_HELP[SPLIT_CMD],
    context_settings={"show_default": True},
)
@click.option(
    "--dataset-csv",
    type=str,
    required=True,
    help="CSV file with dataset metadata",
)
@click.option(
    "--split-prefix",
    type=str,
    required=True,
    help="Prefix used for saving splits.",
)
@click.option(
    "--category-key",
    type=str,
    default="acceptedTaxonKey",
    help="Key used as category id for strified spliting",
)
@click.option(
    "--max-instances",
    type=int,
    default=1000,
    help="Maximun number of instances on training set (and on val/test proportionally)",
)
@with_random_seed
@click.option(
    "--split-by-occurrence",
    type=bool,
    default=True,
    help=(
        "Whether images belonging to the same occurrence should be kept in the same "
        "partition"
    ),
)
@click.option(
    "--test-frac",
    type=float,
    default=0.2,
    help="Fraction of data used for the test set",
)
@click.option(
    "--val-frac",
    type=float,
    default=0.1,
    help="Fraction of data used for the validation set",
)
def split_dataset_command(
    dataset_csv: str,
    split_prefix: str,
    test_frac: float,
    val_frac: float,
    split_by_occurrence: bool,
    category_key: str,
    max_instances: int,
    random_seed: int,
):
    split_dataset(
        dataset_csv=dataset_csv,
        split_prefix=split_prefix,
        test_frac=test_frac,
        val_frac=val_frac,
        split_by_occurrence=split_by_occurrence,
        category_key=category_key,
        max_instances=max_instances,
        random_seed=random_seed,
    )


#
# Create Webdataset Command
#
@click.command(
    name=COMMANDS[WEBDATASET_CMD],
    help=COMMANDS_HELP[WEBDATASET_CMD],
    context_settings={"show_default": True},
)
@click.option(
    "--annotations-csv",
    type=str,
    required=True,
    help="Path to csv file containing the annotations",
)
@with_dataset_path
@click.option(
    "--image-path-column",
    type=str,
    required=True,
    help="CSV column containing image file path",
)
@click.option(
    "--label-column",
    type=str,
    required=True,
    help="CSV column containing image label",
)
@click.option(
    "--shuffle-images",
    type=bool,
    default=True,
    help="Shufle images before to write to tar files",
)
@click.option(
    "--webdataset-patern",
    type=str,
    required=True,
    help="Webdataset output file pattern",
)
@click.option(
    "--category-map-json",
    type=str,
    help=(
        "JSON containing the categories id map. If not provided, the"
        " category map will be infered from annotations csv."
    ),
)
@click.option(
    "--columns-to-json",
    type=str,
    help="List of columns from CSV file to save as metadata in a json file.",
)
@click.option(
    "--max-shard-size",
    type=int,
    default=100 * 1024 * 1024,
    help="Maximun size of each shard",
)
@click.option(
    "--megadetector-results-json",
    type=str,
    help=(
        "Path to json file containing megadetector results. If provided, the"
        " images will be cropped to a squared region around the bbox with"
        " the highest confidence."
    ),
)
@click.option(
    "--resize-min-size",
    type=int,
    help=(
        "Size which the shortest image side will be resized to. If it is not"
        " given, the original image is used withou resizing."
    ),
)
@click.option(
    "--save-category-map-json",
    type=str,
    help=(
        "JSON containing the categories id map. If not provided, the"
        " category map will be infered from annotations csv."
    ),
)
@with_random_seed
def create_webdataset_command(
    annotations_csv: str,
    dataset_dir: str,
    webdataset_patern: str,
    image_path_column: str,
    label_column: str,
    max_shard_size: int,
    shuffle_images: bool,
    resize_min_size: int,
    category_map_json: str,
    save_category_map_json: str,
    columns_to_json: str,
    megadetector_results_json: str,
    random_seed: int,
):
    create_webdataset(
        annotations_csv=annotations_csv,
        dataset_dir=dataset_dir,
        webdataset_pattern=webdataset_patern,
        image_path_column=image_path_column,
        label_column=label_column,
        max_shard_size=max_shard_size,
        shuffle_images=shuffle_images,
        resize_min_size=resize_min_size,
        category_map_json=category_map_json,
        save_category_map_json=save_category_map_json,
        columns_to_json=columns_to_json,
        megadetector_results_json=megadetector_results_json,
        random_seed=random_seed,
    )


# # # # # # # # # # # # # #
# Main CLI configuration  #
# # # # # # # # # # # # # #
class OrderCommands(click.Group):
    """This class is necessary to order the commands the way we want to."""

    def list_commands(self, ctx: click.Context) -> list[str]:
        return list(self.commands)


@click.group(cls=OrderCommands)
def cli():
    """This is the main command line interface for dataset tools."""
    pass


# Following is an automated way to add all functions containing the word `command`
# in their name instead of manually having to add them.
all_objects = globals()
functions = [
    obj for name, obj in all_objects.items() if callable(obj) and "command" in name
]

for command_function in functions:
    cli.add_command(command_function)

if __name__ == "__main__":
    cli()
