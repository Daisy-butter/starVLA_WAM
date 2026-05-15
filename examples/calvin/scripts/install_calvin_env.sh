#!/usr/bin/env bash
# Create a dedicated conda env for CALVIN evaluation (calvin_agent + calvin_env).
# StarVLA training uses the main starVLA env; eval runs a separate client process (see examples/calvin/README.md).
#
# Usage:
#   export CALVIN_BASE=/SSD_DISK/users/wuruihan/sii_starvla/calvin
#   export CONDA_ENV_NAME=calvin
#   bash examples/calvin/scripts/install_calvin_env.sh
#
# Requires: conda, git, CUDA-capable machine for optional EGL rendering upstream recommends.

set -euo pipefail

CALVIN_BASE="${CALVIN_BASE:-/SSD_DISK/users/wuruihan/sii_starvla/calvin_abc_d}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-calvin}"
STARVLA_ROOT="${STARVLA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"

mkdir -p "${CALVIN_BASE}/repos"

if [[ ! -d "${CALVIN_BASE}/repos/calvin/.git" ]]; then
  git clone --recurse-submodules https://github.com/mees/calvin.git "${CALVIN_BASE}/repos/calvin"
fi

CALVIN_SRC="${CALVIN_BASE}/repos/calvin"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH."
  exit 1
fi

# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV_NAME}"; then
  echo "Conda env ${CONDA_ENV_NAME} already exists; run:"
  echo "  conda activate ${CONDA_ENV_NAME} && cd ${CALVIN_SRC} && pip install -e calvin_env -e calvin_models"
  exit 0
fi

conda create -n "${CONDA_ENV_NAME}" python=3.8 -y
conda activate "${CONDA_ENV_NAME}"
cd "${CALVIN_SRC}"
bash install.sh

pip install -U hydra-core omegaconf tyro tqdm termcolor moviepy websocket-client websockets pillow numpy

echo ""
echo "Activate with: conda activate ${CONDA_ENV_NAME}"
echo "Set eval_calvin.sh: export calvin_python=$(conda info --base)/envs/${CONDA_ENV_NAME}/bin/python"
echo "Run eval from StarVLA repo root with PYTHONPATH including: ${STARVLA_ROOT}"
