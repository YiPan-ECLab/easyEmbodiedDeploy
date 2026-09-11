#!/usr/bin/env bash
set -euo pipefail

# Build the isolated OpenPI/JAX environment used by OpenpiTrainer and
# OpenpiDeployer. The main LeRobot/PyTorch environment is intentionally not
# touched: OpenPI and the current LeRobot runtime require different versions.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_DIR="${SCRIPT_DIR}/openpi"
LEROBOT_DIR="${SCRIPT_DIR}/lerobot-openpi"
VENV_DIR="${SCRIPT_DIR}/.venvs/openpi-jax"
LEROBOT_COMMIT="0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"
PYTHON_VERSION="3.11.16"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"

usage() {
    cat <<'EOF'
用法:
  bash thirdParty/setup_openpi_env.sh           创建 OpenPI/JAX 环境
  bash thirdParty/setup_openpi_env.sh --check   只检查环境，不修改任何文件
  bash thirdParty/setup_openpi_env.sh --recreate 删除并重新创建该环境

环境位置:
  thirdParty/.venvs/openpi-jax
EOF
}

die() {
    echo "错误: $*" >&2
    exit 1
}

check_sources() {
    [[ -f "${OPENPI_DIR}/pyproject.toml" ]] || die "缺少 thirdParty/openpi/pyproject.toml。"
    [[ -f "${OPENPI_DIR}/uv.lock" ]] || die "缺少 thirdParty/openpi/uv.lock。"

    if [[ ! -d "${LEROBOT_DIR}/.git" ]]; then
        die "缺少 thirdParty/lerobot-openpi；请先运行创建模式让脚本自动克隆。"
    fi

    local actual_commit
    actual_commit="$(git -C "${LEROBOT_DIR}" rev-parse HEAD)"
    [[ "${actual_commit}" == "${LEROBOT_COMMIT}" ]] || die \
        "lerobot-openpi 提交不匹配：${actual_commit}，需要 ${LEROBOT_COMMIT}。"
}

check_environment() {
    check_sources
    [[ -x "${VENV_DIR}/bin/python" ]] || die "缺少 ${VENV_DIR}/bin/python，请先运行创建模式。"

    # Keep validation deterministic and harmless on a machine that is currently
    # training another job. A separate GPU probe is documented in README.md.
    env -u PYTHONPATH -u AMENT_PREFIX_PATH -u COLCON_PREFIX_PATH \
        JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' \
        "${VENV_DIR}/bin/python" -I - <<'PY'
import importlib.metadata as metadata
import sys

import flax
import jax
import jaxlib
import lerobot
import openpi
import openpi_client
import torch

if sys.version_info[:3] != (3, 11, 16):
    raise RuntimeError(f"Python {sys.version_info[:3]}，需要 (3, 11, 16)")

expected = {
    "jax": "0.5.3",
    "jax-cuda12-plugin": "0.5.3",
    "jax-cuda12-pjrt": "0.5.3",
    "jaxlib": "0.5.3",
    "flax": "0.10.2",
    "torch": "2.7.1",
}
for package, version in expected.items():
    actual = metadata.version(package)
    if actual != version:
        raise RuntimeError(f"{package}=={actual}，需要 {version}")

print("openpi-jax: OK")
print(f"  python={sys.version.split()[0]}")
print(f"  jax={jax.__version__}, jaxlib={jaxlib.__version__}, devices={jax.devices()}")
print(f"  flax={flax.__version__}, torch={torch.__version__}, cuda_available={torch.cuda.is_available()}")
print(f"  lerobot={lerobot.__file__}")
print(f"  openpi={openpi.__file__}")
print(f"  openpi-client={openpi_client.__file__}")
PY
}

clone_lerobot() {
    if [[ -e "${LEROBOT_DIR}" && ! -d "${LEROBOT_DIR}/.git" ]]; then
        die "${LEROBOT_DIR} 已存在但不是 Git 仓库，请移走它后重试。"
    fi
    if [[ ! -d "${LEROBOT_DIR}/.git" ]]; then
        GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/huggingface/lerobot.git "${LEROBOT_DIR}"
        git -C "${LEROBOT_DIR}" checkout --detach "${LEROBOT_COMMIT}"
    fi

    local actual_commit
    actual_commit="$(git -C "${LEROBOT_DIR}" rev-parse HEAD)"
    [[ "${actual_commit}" == "${LEROBOT_COMMIT}" ]] || die \
        "lerobot-openpi 提交不匹配：${actual_commit}，需要 ${LEROBOT_COMMIT}。"
}

if [[ "$#" -gt 1 ]]; then
    usage >&2
    exit 2
fi
mode="${1:-create}"
case "${mode}" in
    create|--check|--recreate) ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

if [[ "${mode}" == "--check" ]]; then
    check_environment
    exit 0
fi

[[ -n "${UV_BIN}" ]] || die "找不到 uv，请先安装：curl -LsSf https://astral.sh/uv/install.sh | sh"
[[ -f "${OPENPI_DIR}/uv.lock" ]] || die "缺少 thirdParty/openpi/uv.lock。"

if [[ "${mode}" == "--recreate" ]]; then
    rm -rf "${VENV_DIR}"
elif [[ -e "${VENV_DIR}" ]]; then
    die "环境已存在：${VENV_DIR}；使用 --check 检查，或使用 --recreate 重建。"
fi

clone_lerobot
unset PYTHONPATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH

"${UV_BIN}" python install "${PYTHON_VERSION}"
"${UV_BIN}" venv --clear --python "${PYTHON_VERSION}" "${VENV_DIR}"

# uv.lock is the source of truth for OpenPI and its CUDA/JAX wheels. --frozen
# prevents setup from silently rewriting the lock file on a new machine.
(
    cd "${OPENPI_DIR}"
    GIT_LFS_SKIP_SMUDGE=1 UV_PROJECT_ENVIRONMENT="${VENV_DIR}" \
        "${UV_BIN}" sync --frozen --no-dev
)

# OpenPI's lock points at the same upstream commit. Installing our local
# checkout editable makes the selected commit explicit and keeps imports
# independent from uv's global Git cache.
"${UV_BIN}" pip install --python "${VENV_DIR}/bin/python" \
    --no-deps --editable "${LEROBOT_DIR}"

check_environment
cat <<EOF

OpenPI/JAX 环境创建完成:
  ${VENV_DIR}
  Python ${PYTHON_VERSION}
  LeRobot commit ${LEROBOT_COMMIT}

运行 OpenPI 训练/部署前可用 CUDA_VISIBLE_DEVICES=0,1 指定 GPU；
安装检查本身使用 CPU，不会占用 GPU。
EOF
