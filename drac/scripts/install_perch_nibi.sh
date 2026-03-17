#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PIP_BIN="${PIP_BIN:-pip}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TF_VERSION="${TF_VERSION:-2.20.0}"

if ! command -v "$PIP_BIN" >/dev/null 2>&1; then
  echo "Error: pip was not found on PATH. Activate the target venv first."
  exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: python was not found on PATH. Activate the target venv first."
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "Error: git is required to build usearch from source."
  exit 1
fi

echo "Installing build prerequisites into the active environment..."
"$PIP_BIN" install --upgrade pip wheel cmake ninja scikit-build-core pybind11

echo "Installing upstream setuptools so pkg_resources is available for tensorflow_hub..."
"$PIP_BIN" install --upgrade --force-reinstall --no-deps --index-url https://pypi.org/simple "setuptools<81"

if ! "$PYTHON_BIN" -c "import pkg_resources" >/dev/null 2>&1; then
  echo "Error: setuptools did not provide pkg_resources in the active environment."
  echo "Try: pip install --upgrade --force-reinstall --no-deps --index-url https://pypi.org/simple 'setuptools<81'"
  exit 1
fi

TMP_ROOT="$(mktemp -d)"
trap 'rm -rf "$TMP_ROOT"' EXIT

echo "Installing SimSIMD from source so USearch links against a compatible build..."
"$PIP_BIN" install --upgrade --force-reinstall --no-deps --no-binary simsimd "simsimd>=6.5,<7"

echo "Building usearch from source (cluster glibc is too old for the published wheel)..."
git clone --depth 1 --recursive https://github.com/unum-cloud/USearch.git "$TMP_ROOT/USearch"
"$PIP_BIN" install --no-build-isolation "$TMP_ROOT/USearch"

echo "Installing TensorFlow ${TF_VERSION} from upstream PyPI so Perch v2 runtime ops are available..."
"$PIP_BIN" --isolated install --upgrade --force-reinstall --no-deps "tensorflow==${TF_VERSION}"

echo "Installing runtime dependencies needed for Perch inference..."
"$PIP_BIN" install \
  "tensorflow-hub>=0.16,<1.0" \
  "absl-py>=1.4,<2" \
  "etils[epath]>=1.5,<2" \
  "imageio>=2.5,<3" \
  "ipywidgets>=8.1,<9" \
  "ml-collections>=0.1.1,<0.2" \
  "notebook>=7.4,<8" \
  "kagglehub>=0.3.13,<0.4" \
  "wandb>=0.15.0"

echo "Installing perch-hoplite without optional pandas[gcp] dependency extras..."
"$PIP_BIN" install --no-deps "perch-hoplite>=1.0.0"

echo "Verifying imports..."
"$PYTHON_BIN" - <<'PY'
import importlib

modules = ["tensorflow", "usearch", "perch_hoplite", "wandb"]
versions = {}
for name in modules:
    module = importlib.import_module(name)
    versions[name] = getattr(module, "__version__", "unknown")

from perch_hoplite.zoo import model_configs
from packaging.version import Version

tf_version = str(versions["tensorflow"]).split("+", 1)[0]
if Version(tf_version) < Version("2.20.0"):
    raise SystemExit(f"TensorFlow {versions['tensorflow']} is too old for Perch v2 runtime")

print("Perch dependencies ready:")
for name in modules:
    print(f"  {name}: {versions[name]}")
print("  perch_hoplite.zoo.model_configs: ok")
PY
