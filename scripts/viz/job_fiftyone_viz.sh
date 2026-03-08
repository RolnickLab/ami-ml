#!/bin/bash
#SBATCH --account=def-drolnick
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --job-name=fiftyone_viz
#SBATCH --output=/project/6068129/melabbas/ami-ml/logs/fiftyone_viz_%j.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hack1996man@gmail.com

NODE=$(hostname)
echo "=============================="
echo "Node: ${NODE}"
echo "Port: 5151"
echo "SSH tunnel: ssh -N -J melabbas@fir.alliancecan.ca -L 5151:localhost:5151 melabbas@${NODE}"
echo "Then open: http://localhost:5151"
echo "=============================="

BASE_DIR="/home/melabbas/projects/def-drolnick/melabbas/ami-ml"
cd "$BASE_DIR"
source .venv/bin/activate

# Bind the FiftyOne server to all interfaces so it's reachable via SSH tunnel
export FIFTYONE_DEFAULT_APP_ADDRESS="0.0.0.0"

python src/dataset_tools/visualize_webdataset.py \
  --webdataset-pattern "data/vermont_butterflies/webdataset/vermont_train_verbatim-{000000..000051}.tar" \
  --category-map-json "data/vermont_butterflies/webdataset/vermont_category_map_verbatim.json" \
  --num-samples 500 \
  --port 5151
