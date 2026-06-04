#!/bin/bash
#SBATCH --account=def-drolnick
#SBATCH --job-name=dwca_inspect_northamerica
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=/project/6068129/melabbas/ami-ml/scripts/dwca_inspect_northamerica_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hack1996man@gmail.com

DWCA="/home/melabbas/projects/def-drolnick/melabbas/ami-ml/data/northamerica_gbif_download_v1/gbif_inat_northamerica_butterflies_106spp_2081177rec_20260227.zip"
DWCA_TOOLS="/home/melabbas/projects/def-drolnick/melabbas/dwca-tools"
REPORT="/home/melabbas/projects/def-drolnick/melabbas/ami-ml/data/northamerica_gbif_download_v1/dwca_inspection_northamerica_butterflies_106spp_2081177rec_image_counts_20260227.txt"

cd "$DWCA_TOOLS"
source .venv/bin/activate

{
echo "############################################################"
echo "# DwCA Inspection Report"
echo "# Archive : $(basename $DWCA)"
echo "# Generated: $(date)"
echo "############################################################"
echo ""

echo "============================================================"
echo "=== SECTION 1: summarize files"
echo "============================================================"
echo ""
dwca-tools summarize files "$DWCA"
echo ""
echo "Completed at $(date)"
echo ""

echo "============================================================"
echo "=== SECTION 2: summarize taxa --image-counts"
echo "============================================================"
echo ""
dwca-tools summarize taxa "$DWCA" \
  --image-counts
echo ""
echo "Completed at $(date)"
echo ""

echo "============================================================"
echo "=== SECTION 3: summarize taxa --group-by verbatimScientificName --show-mismatched-names --image-counts"
echo "============================================================"
echo ""
dwca-tools summarize taxa "$DWCA" \
  --group-by verbatimScientificName \
  --show-mismatched-names \
  --image-counts
echo ""
echo "Completed at $(date)"
echo ""

echo "############################################################"
echo "# End of report"
echo "############################################################"
} | tee "$REPORT"

echo ""
echo "Report saved to: $REPORT"
notify "dwca inspect northamerica done" "3-section inspection report (with image counts) saved to $(basename $REPORT)"
