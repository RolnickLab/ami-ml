import functools

import click

from src.dataset_tools.clean_dataset import clean_dataset
from src.dataset_tools.fetch_images import fetch_images
from src.dataset_tools.verify_images import verify_images

CLEAN_DATASET = "clean-dataset"
FETCH_IMAGES = "fetch-images"
VERIFY_IMAGES = "verify-images"


def with_dwca_file(func):
    @functools.wraps(func)
    @click.option(
        "--dwca-file",
        type=str,
        required=True,
        help="Darwin Core Archive file",
    )
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def with_num_workers(func):
    @functools.wraps(func)
    @click.option(
        "--num-workers",
        type=int,
        default=8,
        help="Number of processes to download in images in parallel",
    )
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


#
# Clean Dataset
#
@click.command(name=CLEAN_DATASET, context_settings={"show_default": True})
@with_dwca_file
@click.option(
    "--verified-data-csv",
    type=str,
    required=True,
    help="CSV file containing verified image info",
)
@click.option(
    "--remove-duplicate-url",
    type=bool,
    default=True,
    help="Whether occurrences with duplicate URLs should be removed.",
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
@click.option(
    "--thumb-size",
    type=int,
    default=64,
    help="Minimum side size to an image not be considered as thumbnail",
)
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
    "--remove-non-adults",
    type=bool,
    default=True,
    help="Whether keeping only occurrences with lifeStage identified as Adult or Imago",
)
@click.option(
    "--life-stage-predictions",
    type=str,
    help=(
        "CSV containing life-stage predictions for images. If provided images without"
        " the GBIF life stage will be filtered based on life stage prediction"
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
# Fetch Images Command
#
@click.command(name=FETCH_IMAGES, context_settings={"show_default": True})
@with_dwca_file
@with_num_workers
@click.option(
    "--dataset-path", type=str, required=True, help="Folder to save images to"
)
@click.option(
    "--cache-path",
    type=str,
    help=(
        "Folder containing cached images. If provided, the script will try copy"
        " images from cache before trying fetch them."
    ),
)
@click.option(
    "--dwca-file",
    type=str,
    required=True,
    help="Darwin Core Archive file",
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
@click.option(
    "--num-images-per-category",
    type=int,
    default=0,
    help=(
        "Number of images to be downloaded for each category. If not provided, all "
        "images will be fetched."
    ),
)
@click.option(
    "--request-timeout",
    type=int,
    default=30,
    help=("Timout for closing urresponsive connections."),
)
@click.option(
    "--random-seed",
    type=int,
    default=42,
    help="Random seed for reproductible experiments",
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


@click.command(name=VERIFY_IMAGES, context_settings={"show_default": True})
@with_dwca_file
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
    "--dataset-path",
    type=str,
    required=True,
    help="Path to directory containing dataset images.",
)
@click.option(
    "--results-csv",
    type=str,
    required=True,
    help="File to save image verification info",
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
# Main CLI configuration
#
@click.group()
def cli():
    """This is the main command line interface for dataset tools."""
    pass


cli.add_command(clean_dataset_command)
cli.add_command(fetch_images_command)
cli.add_command(verify_images_command)

if __name__ == "__main__":
    cli()
