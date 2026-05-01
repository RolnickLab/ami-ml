#!/bin/bash
# Train a ResNet-50 classifier on the global WebDataset.
# Must run after job_build_wds_global_v2.sh completes.
#
# Reads  : /scratch/melabbas/global_wds/{train,val,test}/*.tar + class_map.json
# Writes : /project/6068129/melabbas/ami-ml/models/global_wds/
#
# Usage:
#   sbatch --dependency=afterok:<wds_job_id> job_bq_train.sh
#
#SBATCH --account=def-drolnick_gpu
#SBATCH --job-name=bq_train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/project/6068129/melabbas/ami-ml/scripts/bq_train_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hack1996man@gmail.com

set -euo pipefail

BASE_DIR="/home/melabbas/projects/def-drolnick/melabbas/ami-ml"
WBDS="/scratch/melabbas/global_wds"
MODELS="/project/6068129/melabbas/ami-ml/models/global_wds"

mkdir -p "${MODELS}"

cd "${BASE_DIR}"
source .venv/bin/activate

if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

export WANDB_API_KEY="${WANDB_API_KEY_HACK1996MAN}"

# Dynamically resolve shard counts and num_classes from the WDS output
N_TRAIN=$(find "${WBDS}/train" -name "train-*.tar" | wc -l)
N_VAL=$(find "${WBDS}/val"   -name "val-*.tar"   | wc -l)
N_TEST=$(find "${WBDS}/test" -name "test-*.tar"  | wc -l)
NUM_CLASSES=$(python3 -c "import json; print(len(json.load(open('${WBDS}/class_map.json'))))")

if [[ "${N_TRAIN}" -eq 0 ]]; then
    echo "ERROR: no train shards found in ${WBDS}/train"
    notify "bq_train: FAILED" "No train shards found — did job_build_wds_global_v2 complete?"
    exit 1
fi

TRAIN_PAT="${WBDS}/train/train-{000000..$(printf '%06d' $((N_TRAIN-1)))}.tar"
VAL_PAT="${WBDS}/val/val-{000000..$(printf '%06d' $((N_VAL-1)))}.tar"
TEST_PAT="${WBDS}/test/test-{000000..$(printf '%06d' $((N_TEST-1)))}.tar"

echo "=== bq_train started at $(date) ==="
echo "Node        : $(hostname)"
echo "Train shards: ${N_TRAIN}"
echo "Val shards  : ${N_VAL}"
echo "Test shards : ${N_TEST}"
echo "Num classes : ${NUM_CLASSES}"
echo "Models dir  : ${MODELS}"
echo ""

# Resume from latest checkpoint if available
RESUME_FLAG=""
RESUME_CKPT=$(ls -t "${MODELS}"/*_latest.pt 2>/dev/null | head -1 || true)
if [[ -z "${RESUME_CKPT}" ]]; then
    RESUME_CKPT=$(ls -t "${MODELS}"/*_checkpoint.pt 2>/dev/null | head -1 || true)
fi
if [[ -n "${RESUME_CKPT}" ]]; then
    echo "Resuming from checkpoint: ${RESUME_CKPT}"
    RESUME_FLAG="--resume_from_checkpoint ${RESUME_CKPT}"
else
    echo "No checkpoint found — starting fresh"
fi
echo ""

ami-classification train-model \
    --train_webdataset          "${TRAIN_PAT}" \
    --val_webdataset            "${VAL_PAT}" \
    --test_webdataset           "${TEST_PAT}" \
    --num_classes               "${NUM_CLASSES}" \
    --model_type                resnet50 \
    --image_input_size          128 \
    --model_save_directory      "${MODELS}" \
    --total_epochs              30 \
    --warmup_epochs             2 \
    --early_stopping            100 \
    --learning_rate             0.001 \
    --learning_rate_scheduler   cosine \
    --weight_decay              1e-5 \
    --batch_size                128 \
    --preprocess_mode           torch \
    --mixed_resolution_data_aug true \
    --random_seed               123 \
    --wandb_entity              "hack1996man" \
    --wandb_project             "ai_for_leps" \
    --wandb_run_name            "global_wds_bq_run1" \
    ${RESUME_FLAG}

EXIT_CODE=$?
echo ""
echo "=== bq_train done at $(date) (exit=${EXIT_CODE}) ==="

if [ "${EXIT_CODE}" -eq 0 ]; then
    notify "bq_train: done" \
        "exit=0  classes=${NUM_CLASSES}  train=${N_TRAIN} val=${N_VAL} test=${N_TEST} shards — models in ${MODELS}"
else
    notify "bq_train: FAILED" \
        "exit=${EXIT_CODE} — check bq_train_${SLURM_JOB_ID}.out"
    exit 1
fi
