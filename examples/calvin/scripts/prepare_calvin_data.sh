#!/usr/bin/env bash
# Download CALVIN (original layout), convert to LeRobot v2 for StarVLA training, and stage eval assets.
#
# Usage:
#   export CALVIN_BASE=/SSD_DISK/users/wuruihan/sii_starvla/calvin_abc_d
#   export SPLIT=ABC        # D (~166GB) | ABC (~517GB) | ABCD (~656GB) | debug (~1.3GB)
#   export STARVLA_ROOT=/path/to/starVLA
#   bash examples/calvin/scripts/prepare_calvin_data.sh
#
# After this script:
#   ${CALVIN_BASE}/<split_folder>/        # e.g. task_ABC_D/ after unzip
#   ${CALVIN_BASE}/<split_folder>_lerobot/ # e.g. task_ABC_D_lerobot/ (matches mixtures.py dataset name)
#   ${CALVIN_BASE}/eval_sequences.json    # copy of StarVLA eval sequences
#   ${CALVIN_BASE}/repos/calvin/          # upstream CALVIN code + calvin_models (for eval env)
#
# References:
#   - Dataset: https://github.com/mees/calvin/blob/main/dataset/README.md
#   - LeRobot conversion approach: https://github.com/EmbodiedAI-RoboTron/RoboTron-Mani/tree/lerobot/examples/calvin

set -euo pipefail

STARVLA_ROOT="${STARVLA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
CALVIN_BASE="${CALVIN_BASE:-/SSD_DISK/users/wuruihan/sii_starvla/calvin_abc_d}"
SPLIT="${SPLIT:-debug}"

# Prefer a Python that has StarVLA / numpy (e.g. conda env). Example:
#   export PYTHON=/path/to/conda/envs/starvla/bin/python
PYTHON="${PYTHON:-python3}"
if ! "${PYTHON}" -c "import numpy, pandas, pyarrow; from PIL import Image" 2>/dev/null; then
  echo "ERROR: ${PYTHON} cannot import numpy, pandas, pyarrow, PIL."
  echo "Install them (e.g. pip install numpy pandas pyarrow tyro pillow) or set PYTHON= to your StarVLA interpreter."
  exit 1
fi

mkdir -p "${CALVIN_BASE}/downloads" "${CALVIN_BASE}/repos"

download_and_unzip() {
  local name="$1"
  local url="$2"
  local zip_path="${CALVIN_BASE}/downloads/${name}.zip"
  if [[ ! -f "${zip_path}" ]]; then
    wget -c -O "${zip_path}" "${url}"
  else
    echo "Reusing existing archive: ${zip_path}"
  fi
  unzip -q -o "${zip_path}" -d "${CALVIN_BASE}"
}

case "${SPLIT}" in
  D)
    download_and_unzip "task_D_D" "http://calvin.cs.uni-freiburg.de/dataset/task_D_D.zip"
    ORIG_ROOT="${CALVIN_BASE}/task_D_D"
    ;;
  debug)
    download_and_unzip "calvin_debug_dataset" "http://calvin.cs.uni-freiburg.de/dataset/calvin_debug_dataset.zip"
    ORIG_ROOT="${CALVIN_BASE}/calvin_debug_dataset"
    ;;
  ABC)
    download_and_unzip "task_ABC_D" "http://calvin.cs.uni-freiburg.de/dataset/task_ABC_D.zip"
    ORIG_ROOT="${CALVIN_BASE}/task_ABC_D"
    ;;
  ABCD)
    download_and_unzip "task_ABCD_D" "http://calvin.cs.uni-freiburg.de/dataset/task_ABCD_D.zip"
    ORIG_ROOT="${CALVIN_BASE}/task_ABCD_D"
    ;;
  *)
    echo "SPLIT must be one of: D, debug, ABC, ABCD (got: ${SPLIT})"
    exit 1
    ;;
esac

if [[ ! -d "${ORIG_ROOT}/training" ]]; then
  echo "Expected ${ORIG_ROOT}/training after unzip — check SPLIT / archive layout."
  exit 1
fi

# Optional scene_info fix for ABC split (see mees/calvin README changelog)
if [[ "${SPLIT}" == "ABC" && -d "${ORIG_ROOT}" ]]; then
  echo "If you use an older ABC dump, apply upstream scene_info fix:"
  echo "  cd ${ORIG_ROOT} && wget http://calvin.cs.uni-freiburg.de/scene_info_fix/task_ABC_D_scene_info.zip && unzip -o task_ABC_D_scene_info.zip && rm -f task_ABC_D_scene_info.zip"
fi

# Optional scene_info fix for official D split (see mees/calvin README changelog)
if [[ "${SPLIT}" == "D" && -d "${ORIG_ROOT}" ]]; then
  echo "If you use the official D split, apply scene_info fix when prompted by upstream README:"
  echo "  cd ${ORIG_ROOT} && wget http://calvin.cs.uni-freiburg.de/scene_info_fix/task_D_D_scene_info.zip && unzip -o task_D_D_scene_info.zip && rm -f task_D_D_scene_info.zip"
fi

LEROBOT_OUT="${CALVIN_BASE}/$(basename "${ORIG_ROOT}")_lerobot"
echo "Converting ${ORIG_ROOT} -> ${LEROBOT_OUT} (training split) ..."
"${PYTHON}" "${STARVLA_ROOT}/examples/calvin/scripts/convert_calvin_task_dd_to_lerobot.py" \
  --calvin-root "${ORIG_ROOT}" \
  --output-dir "${LEROBOT_OUT}" \
  --splits training \
  --action-key rel_actions

cp "${STARVLA_ROOT}/examples/calvin/train_files/modality.json" "${LEROBOT_OUT}/meta/modality.json"
echo "Installed meta/modality.json for StarVLA."

# Eval sequences + upstream repo (calvin_agent / conf for eval_calvin.py)
cp -f "${STARVLA_ROOT}/examples/calvin/eval_files/eval_sequences.json" "${CALVIN_BASE}/eval_sequences.json"

if [[ ! -d "${CALVIN_BASE}/repos/calvin/calvin_models" ]]; then
  echo "Cloning mees/calvin (submodules) into ${CALVIN_BASE}/repos/calvin ..."
  git clone --recurse-submodules https://github.com/mees/calvin.git "${CALVIN_BASE}/repos/calvin"
else
  echo "Reusing ${CALVIN_BASE}/repos/calvin"
fi

echo ""
echo "Done."
echo "  Original dataset:     ${ORIG_ROOT}"
echo "  LeRobot (training):   ${LEROBOT_OUT}"
echo "  Eval sequences JSON: ${CALVIN_BASE}/eval_sequences.json"
echo "  Calvin models/conf:   ${CALVIN_BASE}/repos/calvin/calvin_models/conf"
echo ""
echo "Point StarVLA training data_root_dir to: ${CALVIN_BASE}"
echo "  (mixture uses subdirectory: $(basename "${LEROBOT_OUT}"))"
echo ""
echo "For eval_calvin.py defaults, use:"
echo "  --args.dataset_path ${ORIG_ROOT}"
echo "  --args.calvin_config_path ${CALVIN_BASE}/repos/calvin/calvin_models/conf"
echo "  --args.eval_sequences_path ${CALVIN_BASE}/eval_sequences.json"
