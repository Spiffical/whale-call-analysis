#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

IMAGE_URI="docker://tensorflow/tensorflow:2.20.0-gpu"
IMAGE_PATH="${IMAGE_PATH:-${SCRATCH:-$HOME}/whale-call-analysis/containers/tensorflow_2.20.0_gpu.sif}"
VENV_PATH="${VENV_PATH:-${SCRATCH:-$HOME}/whale-call-analysis/venvs/perch2_tf220}"
APPTAINER_MODULE="${APPTAINER_MODULE:-apptainer}"
FORCE_REBUILD="false"

usage() {
  cat <<'USAGE'
Usage:
  bash drac/scripts/prepare_perch2_apptainer_env.sh [options]

Options:
  --image-uri URI             Container source URI (default: docker://tensorflow/tensorflow:2.20.0-gpu)
  --image-path PATH           Target .sif path (default: $SCRATCH/whale-call-analysis/containers/tensorflow_2.20.0_gpu.sif)
  --venv-path PATH            Virtualenv created inside container runtime (default: $SCRATCH/whale-call-analysis/venvs/perch2_tf220)
  --apptainer-module NAME     Module to load before calling apptainer (default: apptainer)
  --force-rebuild             Recreate the venv even if it already exists
  -h, --help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image-uri) IMAGE_URI="$2"; shift 2 ;;
    --image-path) IMAGE_PATH="$2"; shift 2 ;;
    --venv-path) VENV_PATH="$2"; shift 2 ;;
    --apptainer-module) APPTAINER_MODULE="$2"; shift 2 ;;
    --force-rebuild) FORCE_REBUILD="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if command -v module >/dev/null 2>&1 && [[ -n "$APPTAINER_MODULE" ]]; then
  module load "$APPTAINER_MODULE"
fi

if ! command -v apptainer >/dev/null 2>&1; then
  echo "Error: apptainer not found on PATH after loading module '$APPTAINER_MODULE'."
  exit 1
fi

APPTAINER_ENV_SANITIZE=(env -u SSL_CERT_FILE -u REQUESTS_CA_BUNDLE -u CURL_CA_BUNDLE -u PIP_CERT)

mkdir -p "$(dirname "$IMAGE_PATH")"
mkdir -p "$(dirname "$VENV_PATH")"

if [[ ! -f "$IMAGE_PATH" ]]; then
  echo "Pulling container image: $IMAGE_URI"
  "${APPTAINER_ENV_SANITIZE[@]}" apptainer pull "$IMAGE_PATH" "$IMAGE_URI"
else
  echo "Using existing image: $IMAGE_PATH"
fi

if [[ "$FORCE_REBUILD" == "true" ]]; then
  rm -rf "$VENV_PATH"
fi

TMP_ROOT="$(mktemp -d)"
trap 'rm -rf "$TMP_ROOT"' EXIT

USearch_SRC="$TMP_ROOT/USearch"
git clone --depth 1 --recursive https://github.com/unum-cloud/USearch.git "$USearch_SRC"

BIND_PATHS=("$HOME:$HOME" "$(dirname "$VENV_PATH"):$(dirname "$VENV_PATH")" "$TMP_ROOT:$TMP_ROOT")
if [[ -n "${SCRATCH:-}" ]]; then
  BIND_PATHS+=("${SCRATCH}:${SCRATCH}")
fi
BIND_ARG="$(IFS=,; echo "${BIND_PATHS[*]}")"

if [[ ! -x "$VENV_PATH/bin/python" ]]; then
  echo "Creating container-backed virtualenv: $VENV_PATH"
  "${APPTAINER_ENV_SANITIZE[@]}" apptainer exec --bind "$BIND_ARG" "$IMAGE_PATH" python -m venv "$VENV_PATH"
else
  echo "Using existing container-backed virtualenv: $VENV_PATH"
fi

echo "Installing Perch v2 dependencies inside container-backed virtualenv..."
"${APPTAINER_ENV_SANITIZE[@]}" apptainer exec --bind "$BIND_ARG" "$IMAGE_PATH" bash -lc "
  set -euo pipefail
  export PIP_CONFIG_FILE=/dev/null
  export PIP_DISABLE_PIP_VERSION_CHECK=1
  source \"$VENV_PATH/bin/activate\"
  python -m pip install --isolated --upgrade pip wheel cmake ninja scikit-build-core pybind11
  python -m pip install --isolated --upgrade --force-reinstall --no-deps --index-url https://pypi.org/simple 'setuptools<81'
  python -m pip install --isolated --upgrade --force-reinstall --no-deps --no-binary simsimd 'simsimd>=6.5,<7'
  python -m pip install --isolated --no-build-isolation --no-deps \"$USearch_SRC\"
  python -m pip install --isolated \
    'tensorflow-hub>=0.16,<1.0' \
    'absl-py>=1.4,<2' \
    'etils[epath]>=1.5,<2' \
    'imageio>=2.5,<3' \
    'ipywidgets>=8.1,<9' \
    'ml-collections>=0.1.1,<0.2' \
    'notebook>=7.4,<8' \
    'kagglehub>=0.3.13' \
    'wandb>=0.15.0' \
    'packaging>=24,<26'
  python -m pip install --isolated --no-deps 'perch-hoplite>=1.0.0'
"

echo "Running a Perch v2 import smoke test inside the container..."
"${APPTAINER_ENV_SANITIZE[@]}" apptainer exec --bind "$BIND_ARG" "$IMAGE_PATH" bash -lc "
  set -euo pipefail
  source \"$VENV_PATH/bin/activate\"
  python - <<'PY'
import numpy as np
import tensorflow as tf
from perch_hoplite.zoo import model_configs

model = model_configs.load_model_by_name('perch_v2_cpu')
audio = np.zeros((1, int(round(model.sample_rate * model.window_size_s))), dtype=np.float32)
model.batch_embed(audio)
print('Perch v2 smoke ok', tf.__version__, model.sample_rate, model.window_size_s)
PY
"

cat <<EOF
Container-backed Perch v2 environment is ready.
  image: $IMAGE_PATH
  venv:  $VENV_PATH

Use these with the submit script:
  --container-image "$IMAGE_PATH"
  --container-venv-path "$VENV_PATH"
EOF
