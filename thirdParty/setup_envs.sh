#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PIPER_ENV="$SCRIPT_DIR/.venvs/piper-home"
LEROBOT_ENV="$SCRIPT_DIR/.venvs/lerobot-pi05"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"

usage() {
    cat <<'EOF'
Usage:
  bash thirdParty/setup_envs.sh          Create both environments on a fresh clone
  bash thirdParty/setup_envs.sh --check  Check existing environments without changing them
  bash thirdParty/setup_envs.sh --recreate
                                         Clear and rebuild both environments
EOF
}

check_envs() {
    env -u PYTHONPATH -u AMENT_PREFIX_PATH -u COLCON_PREFIX_PATH \
        "$PIPER_ENV/bin/python" -I - <<'PY'
import h5py
import numpy
import cv2
import can
import zmq
import pyorbbecsdk
from pyAgxArm import ArmModel

print("piper-home: OK")
print(f"  numpy={numpy.__version__}, h5py={h5py.__version__}, opencv={cv2.__version__}")
PY

    env -u PYTHONPATH -u AMENT_PREFIX_PATH -u COLCON_PREFIX_PATH \
        "$LEROBOT_ENV/bin/python" -I - <<'PY'
import torch
import torchvision
import lerobot
import flask
import h5py
import zmq

print("lerobot-pi05: OK")
print(f"  torch={torch.__version__}, torchvision={torchvision.__version__}")
print(f"  torch CUDA runtime={torch.version.cuda}, CUDA available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU={torch.cuda.get_device_name(0)}")
    x = torch.ones((256, 256), device="cuda")
    torch.mm(x, x)
    torch.cuda.synchronize()
    print("  CUDA matrix multiply: OK")
print(f"  compiled architectures={torch.cuda.get_arch_list()}")
PY
}

mode="${1:-create}"
case "$mode" in
    create|--check|--recreate) ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

if [[ "$mode" == "--check" ]]; then
    check_envs
    exit 0
fi

if [[ -z "$UV_BIN" ]]; then
    echo "uv was not found in PATH. Install it first: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

if [[ "$mode" == "create" ]] && \
   { [[ -e "$PIPER_ENV/bin/python" ]] || [[ -e "$LEROBOT_ENV/bin/python" ]]; }; then
    echo "An environment already exists under $SCRIPT_DIR/.venvs." >&2
    echo "Use --check to verify it or --recreate to replace both environments." >&2
    exit 1
fi

# ROS setup scripts commonly add Python 3.10 packages to PYTHONPATH. They must
# not leak into the isolated Python 3.12 LeRobot environment.
unset PYTHONPATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH

"$UV_BIN" python install 3.10.12 3.12.14
"$UV_BIN" venv --clear --python 3.10.12 "$PIPER_ENV"
"$UV_BIN" pip install --python "$PIPER_ENV/bin/python" \
    --no-deps \
    -r "$SCRIPT_DIR/environment-locks/piper-home.runtime-lock.txt"
"$UV_BIN" pip install --python "$PIPER_ENV/bin/python" \
    --no-deps --editable "$SCRIPT_DIR/pyAgxArm"

"$UV_BIN" venv --clear --python 3.12.14 "$LEROBOT_ENV"
"$UV_BIN" pip install --python "$LEROBOT_ENV/bin/python" \
    --no-deps \
    --index-url https://download.pytorch.org/whl/cu128 \
    -r "$SCRIPT_DIR/environment-locks/lerobot-pytorch-cu128.lock.txt"
"$UV_BIN" pip install --python "$LEROBOT_ENV/bin/python" \
    --no-deps \
    -r "$SCRIPT_DIR/environment-locks/lerobot-pi05.runtime-lock.txt"
"$UV_BIN" pip install --python "$LEROBOT_ENV/bin/python" \
    --no-deps --editable "$SCRIPT_DIR/lerobot"
"$UV_BIN" pip install --python "$LEROBOT_ENV/bin/python" \
    --no-deps --editable "$SCRIPT_DIR/pyAgxArm"

check_envs
cat <<EOF

Environment setup is complete:
  $PIPER_ENV
  $LEROBOT_ENV

Project root:
  $PROJECT_ROOT
EOF
