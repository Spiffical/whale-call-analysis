#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PIP_BIN="${PIP_BIN:-pip}"
PYTHON_BIN="${PYTHON_BIN:-python}"

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
"$PIP_BIN" install --upgrade pip setuptools wheel cmake ninja scikit-build-core pybind11

TMP_ROOT="$(mktemp -d)"
trap 'rm -rf "$TMP_ROOT"' EXIT

echo "Building usearch from source (cluster glibc is too old for the published wheel)..."
git clone --depth 1 --recursive https://github.com/unum-cloud/USearch.git "$TMP_ROOT/USearch"
"$PIP_BIN" install --no-build-isolation "$TMP_ROOT/USearch"

echo "Installing Perch training dependencies from the repo requirements..."
"$PIP_BIN" install -r "$REPO_ROOT/requirements-perch.txt"

echo "Verifying imports..."
"$PYTHON_BIN" - <<'PY'
import importlib

modules = ["tensorflow", "usearch", "perch_hoplite", "wandb"]
versions = {}
for name in modules:
    module = importlib.import_module(name)
    versions[name] = getattr(module, "__version__", "unknown")

print("Perch dependencies ready:")
for name in modules:
    print(f"  {name}: {versions[name]}")
PY
