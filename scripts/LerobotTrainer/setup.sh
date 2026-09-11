#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
AUTO_PORT_START="${LEROBOT_TRAINER_PORT_START:-8891}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "找不到 Python：${PYTHON_BIN}" >&2
  exit 1
fi

PORT="auto"
HOST="127.0.0.1"
FORWARD_ARGS=()
while (($#)); do
  case "$1" in
    --port)
      if (($# < 2)); then
        echo "--port 需要端口号或 auto" >&2
        exit 2
      fi
      PORT="$2"
      shift 2
      ;;
    --port=*)
      PORT="${1#--port=}"
      shift
      ;;
    --host)
      if (($# < 2)); then
        echo "--host 需要监听地址" >&2
        exit 2
      fi
      HOST="$2"
      FORWARD_ARGS+=("$1" "$2")
      shift 2
      ;;
    --host=*)
      HOST="${1#--host=}"
      FORWARD_ARGS+=("$1")
      shift
      ;;
    *)
      FORWARD_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "${PORT}" == "auto" ]]; then
  PORT="$("${PYTHON_BIN}" - "${HOST}" "${AUTO_PORT_START}" <<'PY'
import socket
import sys

host = sys.argv[1]
start = int(sys.argv[2])
if not 1 <= start <= 65535:
    raise SystemExit("LEROBOT_TRAINER_PORT_START 必须在 1–65535 之间")
for port in range(start, min(start + 100, 65536)):
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    try:
        with socket.socket(family, socket.SOCK_STREAM) as probe:
            probe.bind((host, port))
    except OSError:
        continue
    print(port)
    break
else:
    raise SystemExit(f"从 {start} 起连续 100 个端口都已占用")
PY
)"
  echo "自动选择空闲端口：${PORT}（扫描起点 ${AUTO_PORT_START}）"
elif [[ ! "${PORT}" =~ ^[0-9]+$ ]] || ((PORT < 1 || PORT > 65535)); then
  echo "--port 必须是 1–65535 或 auto" >&2
  exit 2
fi

cd "${REPO_ROOT}"
export PYTHONDONTWRITEBYTECODE=1
exec "${PYTHON_BIN}" "${SCRIPT_DIR}/web_server.py" --port "${PORT}" "${FORWARD_ARGS[@]}"
