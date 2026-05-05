#!/usr/bin/env bash
# Bootstrap ami-workspace-02-gpu (or any equivalent) for leps localizer training.
#
# Idempotent — safe to re-run. Each step skips if already satisfied.
#
# Usage (from the VM):
#   git clone -b feat/leps-localizer-training git@github.com:RolnickLab/ami-ml.git ~/ami-ml
#   cd ~/ami-ml && bash research/leps_localizer/scripts/setup_workspace_vm.sh
#
# What this does:
#   1. sanity-check uv, git, GPU
#   2. install ami-ml in a uv venv with the detection extra
#   3. ensure /mnt/s3 (old-cloud ami-trainingdata) is mounted via systemd
#   4. (re)create /mnt/squash-0 if the unit was disabled
#   5. install wandb + prompt user to login (interactive)
#   6. emit a one-line "ready" summary
#
# Notes:
#   - Run as the `debian` user on the VM (not root).
#   - The ami-ml repo's `pyproject.toml` must declare a `detection` optional
#     extra (see Phase 3 plan task). If that extra doesn't exist yet, the
#     `uv sync` step will fail loudly — that's expected during plan rollout.

set -euo pipefail

REPO_ROOT="${HOME}/ami-ml"
VENV_DIR="${REPO_ROOT}/.venv"

err() { printf 'setup_workspace_vm.sh: \033[31m%s\033[0m\n' "$*" >&2; exit 1; }
info() { printf 'setup_workspace_vm.sh: %s\n' "$*"; }

# --- 1. sanity-check toolchain ----------------------------------------------
[[ -d "${REPO_ROOT}" ]] || err "expected ${REPO_ROOT} (ami-ml clone) to exist"
command -v uv >/dev/null || err "uv not on PATH (was pre-installed in workspace baseline)"
command -v git >/dev/null || err "git missing"
nvidia-smi -L >/dev/null 2>&1 || err "nvidia driver not reachable — is this the GPU box?"
info "uv $(uv --version), git $(git --version | awk '{print $3}'), GPU $(nvidia-smi -L | head -1)"

# --- 2. uv sync with detection extra ----------------------------------------
cd "${REPO_ROOT}"
if [[ ! -d "${VENV_DIR}" ]]; then
    info "creating venv via uv sync"
fi
if uv sync --extra detection 2>&1 | tee /tmp/uv-sync.log; then
    info "uv sync ok"
else
    err "uv sync failed — see /tmp/uv-sync.log"
fi

# --- 3. confirm /mnt/s3 mount (old-cloud ami-trainingdata) ------------------
if mountpoint -q /mnt/s3; then
    info "/mnt/s3 already mounted (old-cloud ami-trainingdata)"
else
    info "/mnt/s3 not mounted — starting ami-trainingdata-s3.service"
    sudo systemctl start ami-trainingdata-s3.service
    sleep 3
    mountpoint -q /mnt/s3 || err "/mnt/s3 still not mounted after systemctl start"
fi

# --- 4. confirm /mnt/squash-0 (one of the global_butterflies sqfs shards) ---
if mountpoint -q /mnt/squash-0; then
    info "/mnt/squash-0 already mounted"
else
    info "/mnt/squash-0 not mounted — invoking helper if present"
    if [[ -x ${HOME}/ami-devops/scripts/mount-squash-shard.sh ]]; then
        sudo ${HOME}/ami-devops/scripts/mount-squash-shard.sh 0
    else
        info "no helper script — skipping (not strictly required for stage-3 training)"
    fi
fi

# --- 5. wandb -----------------------------------------------------------------
if [[ ! -f "${VENV_DIR}/bin/wandb" ]]; then
    info "wandb not in venv — adding to project group 'detection' is preferred,"
    info "but ad-hoc tool install also fine:"
    uv tool install wandb >/dev/null 2>&1 || true
fi
if [[ ! -f "${HOME}/.netrc" ]] || ! grep -q "machine api.wandb.ai" "${HOME}/.netrc" 2>/dev/null; then
    cat <<'EOF' >&2

  >>> wandb is not logged in.

  Run:    wandb login

  (paste your API key from https://wandb.ai/settings)

EOF
fi

# --- 6. summary ---------------------------------------------------------------
info "READY"
info "  repo:      ${REPO_ROOT}"
info "  venv:      ${VENV_DIR}"
info "  data:      /mnt/s3/ (old-cloud, ro), /mnt/squash-0/ (sqfs ro), /mnt/ (246GB ephemeral rw)"
info "  next:      cd ${REPO_ROOT} && uv run python -c \"import torch; print(torch.cuda.is_available())\""
