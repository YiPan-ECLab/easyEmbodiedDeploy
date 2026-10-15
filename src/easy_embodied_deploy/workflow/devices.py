"""Workflow device selection uses physical GPU indices from nvidia-smi."""

import copy
import os
import re

from ..system import gpu_status


def normalize_device(value):
    if not isinstance(value, str):
        raise ValueError('推理设备须为 cpu 或 cuda:N（例如 cuda:0）')
    value = value.strip().lower()
    if value == 'cpu':
        return value
    match = re.fullmatch(r'cuda(?::([0-9]+))?', value)
    if not match:
        raise ValueError('推理设备须为 cpu 或 cuda:N（例如 cuda:0）')
    return f'cuda:{int(match.group(1) or 0)}'


def worker_launch(config):
    """Pin either backend to the chosen physical GPU, with local index zero."""
    worker = copy.deepcopy(config)
    env = os.environ.copy()
    if config['backend'] == 'mock':
        return worker, env
    device = normalize_device(config.get('device', 'cuda:0'))
    if device == 'cpu':
        env.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu')
        worker['device'] = 'cpu'
    else:
        index = int(device.split(':')[1])
        gpu = next((g for g in gpu_status() if g['index'] == index), None)
        if gpu is None:
            raise ValueError(f'未检测到物理 GPU {index}；请刷新 GPU 列表并选择可用设备')
        # UUID avoids CUDA/NVML enumeration differences and inherited masks.
        env.update(CUDA_VISIBLE_DEVICES=gpu['uuid'], JAX_PLATFORMS='cuda')
        worker['device'] = 'cuda:0'
    worker['requested_device'] = device
    # Independent JAX workers must not each reserve most of the GPU up front.
    env['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    return worker, env
