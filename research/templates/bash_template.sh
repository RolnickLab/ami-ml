#! /bin/bash

set -o errexit  # Exit on error
set -o pipefail # Exit when a command in a pipeline fails
set -o nounset  # Exit when using undeclared variables

################################################################################
# This is an example of a bash script that runs a python script.
# It loads the environment variables using source instead of dotenv
# inside the python script.
#
# Run this from the projet root directory.
# ./research/templates/bash_template.sh
################################################################################

# Load the environment variables outside of python script
set -o allexport
source .env set
set +o allexport

echo "Your logs directory is: ${OUTPUT_LOGS_DIR}"

python research/templates/script_template.py
