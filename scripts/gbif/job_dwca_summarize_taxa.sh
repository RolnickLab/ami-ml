#!/bin/bash
#SBATCH --account=def-drolnick
#SBATCH --job-name=dwca_summarize_taxa
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=/project/6068129/melabbas/ami-ml/scripts/dwca_summarize_taxa_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hack1996man@gmail.com

DWCA="/home/melabbas/projects/def-drolnick/melabbas/ami-ml/data/vermont_gbif_download_v2/gbif_inat_vermont_butterflies_106spp_2479479rec_20260225.zip"
DWCA_TOOLS="/home/melabbas/projects/def-drolnick/melabbas/dwca-tools"

cd "$DWCA_TOOLS"
source .venv/bin/activate

echo "=== dwca-tools summarize taxa ==="
echo "Archive: $DWCA"
echo "Started at $(date)"
echo ""

dwca-tools summarize taxa "$DWCA"

echo ""
echo "Completed at $(date)"
