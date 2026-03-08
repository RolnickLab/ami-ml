#!/bin/bash
# =============================================================================
# launch_fiftyone.sh — Submit and monitor a FiftyOne visualization job
#
# Usage:
#   bash scripts/launch_fiftyone.sh
#
# What it does:
#   1. Submits job_fiftyone_viz.sh to SLURM
#   2. Waits for the job to start running
#   3. Waits for FiftyOne to finish loading samples
#   4. Prints the exact SSH tunnel command to run on your laptop
#
# Notes:
#   - The SLURM job (job_fiftyone_viz.sh) runs for up to 2 hours
#   - To visualize vermont_species instead of vermont_butterflies,
#     edit job_fiftyone_viz.sh and update --webdataset-pattern and
#     --category-map-json before running this script
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/job_fiftyone_viz.sh"
LOG_DIR="${SCRIPT_DIR}/../logs"

mkdir -p "$LOG_DIR"

echo "Submitting FiftyOne visualization job..."
SUBMIT_OUT=$(sbatch "$JOB_SCRIPT")
JOB_ID=$(echo "$SUBMIT_OUT" | awk '{print $NF}')
echo "Submitted job ${JOB_ID}"

# Wait for the job to start running
echo "Waiting for job to start..."
while true; do
    STATUS=$(squeue -j "$JOB_ID" -h -o "%T %N" 2>/dev/null)
    STATE=$(echo "$STATUS" | awk '{print $1}')
    NODE=$(echo "$STATUS" | awk '{print $2}')

    if [[ "$STATE" == "RUNNING" ]]; then
        echo "Job is RUNNING on node: ${NODE}"
        break
    elif [[ -z "$STATUS" ]]; then
        echo "ERROR: Job ${JOB_ID} disappeared from queue (may have failed immediately)"
        exit 1
    fi
    sleep 5
done

LOG_FILE="${LOG_DIR}/fiftyone_viz_${JOB_ID}.log"

# Wait for FiftyOne server to be ready
echo "Waiting for FiftyOne to load samples and start server..."
READY=false
for i in $(seq 1 60); do
    if [[ -f "$LOG_FILE" ]]; then
        # Check if server is up (no more "Could not connect" and has the remote session banner)
        if grep -q "remote sessions" "$LOG_FILE" 2>/dev/null && \
           ! grep -q "Could not connect" "$LOG_FILE" 2>/dev/null; then
            READY=true
            break
        fi
        # Also check if the curl endpoint responds
        if curl -s --connect-timeout 2 "http://${NODE}:5151/fiftyone" 2>/dev/null | grep -q "version"; then
            READY=true
            break
        fi
    fi
    sleep 10
done

if [[ "$READY" == "false" ]]; then
    echo "WARNING: FiftyOne may not have started cleanly. Check log: ${LOG_FILE}"
    echo "You can still try the tunnel — it may just need more time."
fi

echo ""
echo "============================================================"
echo "  FiftyOne is running!"
echo "  Job:  ${JOB_ID}  (up to 2 hours)"
echo "  Node: ${NODE}"
echo "  Log:  ${LOG_FILE}"
echo ""
echo "  Run this on your LAPTOP to open the UI:"
echo ""
echo "    ssh -N -J melabbas@fir.alliancecan.ca -L 5151:localhost:5151 melabbas@${NODE}"
echo ""
echo "  Then open: http://localhost:5151"
echo "============================================================"
echo ""
echo "To cancel the job early: scancel ${JOB_ID}"
