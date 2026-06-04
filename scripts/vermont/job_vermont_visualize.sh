#!/bin/bash
#SBATCH --account=def-drolnick
#SBATCH --job-name=vermont_visualize
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=vermont_visualize_%j.out

BASE_DIR="/home/melabbas/projects/def-drolnick/melabbas/ami-ml"
WBDS="$BASE_DIR/data/vermont_butterflies/webdataset"
PORT=5151

cd "$BASE_DIR"
source .venv/bin/activate

# Install viz extras if fiftyone not yet available
python -c "import fiftyone" 2>/dev/null || uv pip install -e ".[viz]"

echo "Node: $(hostname)"
echo "SSH tunnel: ssh -L ${PORT}:$(hostname):${PORT} melabbas@fir.alliancecan.ca"

python src/dataset_tools/visualize_webdataset.py \
  --webdataset-pattern "$WBDS/vermont_train_verbatim-{000000..000051}.tar" \
  --category-map-json "$WBDS/vermont_category_map_verbatim.json" \
  --num-samples 500 \
  --port $PORT
