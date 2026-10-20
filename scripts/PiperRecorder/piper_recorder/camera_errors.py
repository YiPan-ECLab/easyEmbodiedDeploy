"""Best-effort, read-only diagnostics for camera startup failures."""

import os
import re
from pathlib import Path


def device_holders(serial, error, *, sysfs=Path('/sys'), proc=Path('/proc')):
    """Find USB/video handles for this device, without opening the device."""
    devices = set()
    match = re.search(r'Path:\s*(\d+-[\d.]+)', error)
    usb_root = sysfs / 'bus/usb/devices'
    if match:
        # SDK paths may end with '-<interface>.<index>'.
        devices.add(usb_root / match[1])
    for device in usb_root.glob('*'):
        try:
            if (device / 'serial').read_text().strip() == serial:
                devices.add(device)
        except OSError:
            continue
    nodes = set()
    for device in devices:
        try:
            bus = int((device / 'busnum').read_text())
            number = int((device / 'devnum').read_text())
            nodes.add(f'/dev/bus/usb/{bus:03d}/{number:03d}')
            for video in (sysfs / 'class/video4linux').glob('video*'):
                if device.resolve() in (video / 'device').resolve().parents:
                    nodes.add(f'/dev/{video.name}')
        except (OSError, ValueError):
            continue
    holders = []
    for process in proc.glob('[0-9]*'):
        try:
            for fd in (process / 'fd').iterdir():
                try:
                    target = os.readlink(fd)
                except OSError:
                    continue
                if target in nodes:
                    command = (process / 'cmdline').read_bytes().split(b'\0')
                    # Include executable/script names, never model prompts or full arguments.
                    names = [Path(arg.decode(errors='replace')).name for arg in command
                             if arg and (arg == command[0] or arg.endswith(b'.py'))]
                    holders.append(f'PID {process.name} ({" ".join(names)})')
                    break
        except OSError:
            continue
    return holders


def camera_start_error(role, serial, exc):
    messages, seen = [], set()
    cause = exc
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        messages.append(str(cause))
        cause = cause.__cause__ or cause.__context__
    raw = ' → '.join(messages)
    lower = raw.lower()
    if re.search(r'return code:\s*-6\b', lower) or 'busy' in lower:
        explanation = ('USB 接口忙或已被占用。工作流执行结束/停止不会断开设备，'
                       '请在工作流页面点击「断开设备」，并停止其他采集、标定或相机预览后重试。')
    elif re.search(r'return code:\s*-3\b', lower) or 'permission' in lower or 'access denied' in lower:
        explanation = 'USB 访问权限不足，请检查相机 udev 规则和当前用户的设备权限。'
    elif re.search(r'return code:\s*-4\b', lower) or 'no device' in lower:
        explanation = 'USB 设备已断开或不可用，请检查相机连接与供电。'
    else:
        explanation = '请根据原始错误检查相机连接、序列号、流配置及 SDK 支持。'
    try:
        holders = device_holders(serial, raw)
    except Exception:
        holders = []  # Diagnostic failures must not hide the SDK error.
    ownership = ('检测到持有该设备句柄的进程：' + '、'.join(holders) + '。'
                 if holders else '未查到可见的设备占用进程（不代表设备一定空闲）。')
    return RuntimeError(f'相机 {role}（序列号 {serial}）启动失败：{explanation} '
                        f'{ownership} 原始错误：{raw}')
