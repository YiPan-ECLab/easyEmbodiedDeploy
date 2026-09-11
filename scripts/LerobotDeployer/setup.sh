#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "找不到 Python：${PYTHON_BIN}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" "${SCRIPT_DIR}/web_server.py" "$@"
