"""
This is an example of generalizing/sharing configurations in a python file,
as importable variables.

This is a great way to replace relative paths and since the path is determined
programmatically, it will also work for everybody without any more configuration

This type of file should be committed if used like described above.

However, this could also be used like the `bash_template.sh` and the `.env` file
used by the `script_template.py`, in cases where it contains variables that are
only relevant to a personal environment, or different instances of a deployed
application. In that case, the `config.py` would not be committed to the repository,
only a template with placeholder values that are intended to be replaced.
"""

import pathlib

# RESEARCH_TEMPLATE_ROOT would be equal to /PATH/TO/ami-ml/research/template/, because
# this file is located here : /PATH/TO/ami-ml/research/template/config_example.py,
# hence the parent folder of this file.
RESEARCH_TEMPLATE_ROOT = pathlib.Path(__file__).resolve().parent

# RESEARCH_ROOT would be equal to /PATH/TO/ami-ml/research/
RESEARCH_ROOT = RESEARCH_TEMPLATE_ROOT.parent

# PROJECT_ROOT would be equal to /PATH/TO/ami-ml
PROJECT_ROOT = RESEARCH_ROOT.parent

# We can even define a path to the `assets/` directory, or any future folder that
# could be of interest
ASSETS_PATH = PROJECT_ROOT / "assets"

# It's even possible to define project related paths that are still the same for
# everyone on the projet, say a path to a network dataset
AMI_MILA_EXAMPLE_DATASET_PATH = pathlib.Path("/network/datasets/ami")

# From there, these constants can be imported into other python scripts. This is an
# approach that is very similar to how Django manages it's configurations
# see : https://docs.djangoproject.com/en/5.0/topics/settings/

# Other approaches can still be used, like .ini files with `configparser`
# see : https://docs.python.org/3/library/configparser.html

# We could also get very creative and create a data class that would load, parse
# and manage imported variables from an `.env` file, but this would be overkill
# for the present use case.


# You can execute this file to get a preview of the path
# resolution
if __name__ == "__main__":
    print(RESEARCH_TEMPLATE_ROOT)
    print(RESEARCH_ROOT)
    print(PROJECT_ROOT)
    print(ASSETS_PATH)
    print(AMI_MILA_EXAMPLE_DATASET_PATH)
