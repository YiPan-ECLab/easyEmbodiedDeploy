"""Isolated inference process: no CAN, camera or motion-control interfaces."""

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import replace
from multiprocessing.connection import Listener
from pathlib import Path

import numpy as np

from ..module_http import ROOT


def progress(stage, **extra):
    print("WORKFLOW " + json.dumps(dict(stage=stage, **extra), ensure_ascii=False), flush=True)


def load(config):
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts/LerobotDeployer"))
    from scripts.OpenpiDeployer.runtime import PolicyChunkSampler, RTCSettings, validate_manifest
    from ..contracts import checkpoint_action_mode
    from ..target_runtime import TargetSampler, TorchPolicyAdapter, checkpoint_pose_contract
    checkpoint = Path(config["checkpoint"]).expanduser()
    checkpoint = (ROOT / checkpoint).resolve() if not checkpoint.is_absolute() else checkpoint.resolve()
    mode = config["action_mode"]
    use_rtc = config.get("inference_mode", "rtc" if config.get("rtc", True) else "sync") == "rtc"
    rtc = RTCSettings(mode=config.get("rtc_mode", "guided"), horizon=config.get("rtc_horizon",20),
                      queue_threshold=config.get("rtc_queue_threshold",40), policy_hz=config.get("policy_hz",10),
                      schedule=config.get("rtc_schedule","exp"),
                      max_guidance_weight=config.get("rtc_max_guidance_weight",5),
                      mask_grippers=config.get("rtc_mask_grippers",True)) if use_rtc else None
    progress("loading", detail="读取 checkpoint 与模型权重")
    if config["backend"] == "lerobot":
        from LerobotDeployer import load_policy
        from lerobot.policies.rtc.configuration_rtc import RTCConfig
        checkpoint_action_mode(checkpoint, mode)
        policy, pre, post, _ = load_policy(checkpoint, config.get("device", "cuda:0"),
                                          RTCConfig(mode=rtc.mode, execution_horizon=rtc.horizon) if rtc else None)
        trained_delay = int(getattr(policy.config, "rtc_training_max_delay", 0) or 0)
        if int(getattr(policy.config, "chunk_size", 50)) != 50:
            raise ValueError("当前共享动作队列要求 checkpoint chunk_size=50")
        reuse_steps = int(getattr(policy.config, "n_action_steps", 50))
        native = TorchPolicyAdapter(policy, pre, post, config.get("device", "cuda:0"))
    else:
        from scripts.OpenpiDeployer.runtime import resolve_checkpoint, load_training_manifest, load_policy
        checkpoint = resolve_checkpoint(checkpoint)
        _, manifest = load_training_manifest(checkpoint)
        if manifest["config"]["action_mode"] != mode:
            raise ValueError("checkpoint 动作模式与工作流模型配置不一致")
        trained_delay = int(manifest["config"].get("rtc_max_delay",0) or 0)
        reuse_steps = int(manifest["config"].get("n_action_steps",50))
        os.environ["OPENPI_RTC_SAMPLE_MODE"] = "train_time" if config.get("rtc_mode") == "trained" else "guidance"
        os.environ["OPENPI_TT_RTC_MASK_GRIPPER"] = "1" if config.get("rtc_mask_grippers",True) else "0"
        native = load_policy(checkpoint, manifest["config"], task=config.get("task", ""), num_steps=config.get("sample_steps",10))
    if rtc:
        validate_manifest({"config":dict(action_mode=mode,rtc_max_delay=trained_delay)},
                          action_mode=mode,inference_mode="rtc",rtc_mode=rtc.mode,
                          rtc_horizon=rtc.horizon,queue_threshold=rtc.queue_threshold)
        rtc = replace(rtc, training_max_delay=trained_delay)
    reuse_steps = config.get("n_action_steps",0) or reuse_steps
    if not 1 <= reuse_steps <= 50:
        raise ValueError("同步复用动作数须在 checkpoint 的 50 步 chunk 范围内")
    contract = checkpoint_pose_contract(checkpoint, mode, mount_id=config.get("mount_id", ""))
    sampler = PolicyChunkSampler(native, config.get("task", ""), rtc)
    progress("warming", detail="预热推理（含 RTC），此时不连接机械臂")
    sampler.warmup()
    return TargetSampler(sampler, mode, contract), contract, dict(training_max_delay=trained_delay, action_steps=reuse_steps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(args.config)
    sampler = contract = None
    metadata = dict(training_max_delay=0, action_steps=config.get("n_action_steps",0) or 50)
    try:
        if config["backend"] == "mock":
            progress("loading", detail="模拟模型加载")
            time.sleep(float(config.get("load_delay", .1)))
            if config.get("fail_load"):
                raise ValueError("模拟加载失败")
            progress("warming", detail="模拟预热")
        else:
            sampler, contract, metadata = load(config)
        with Listener(("127.0.0.1", 0), authkey=bytes.fromhex(os.environ["WORKFLOW_AUTH"])) as server:
            progress("ready", port=server.address[1], contract=contract, inference=metadata, detail="权重与预热已完成")
            while True:
                with server.accept() as connection:
                    try:
                        request = connection.recv()
                        if request["op"] == "close":
                            connection.send({"ok": True})
                            break
                        if request["op"] != "infer":
                            raise ValueError("未知推理请求")
                        obs = request["observation"]
                        if sampler is None:
                            time.sleep(float(config.get("infer_delay", .03)))
                            # A simulated hold policy; never generates hardware targets.
                            if config["action_mode"] in ("absolute_joint", "delta_joint"):
                                anchor = obs["observation.state"]
                            else:
                                pose, q = obs["observation.flange_pose"], obs["observation.state"]
                                anchor = np.r_[pose[:6], q[6], pose[6:], q[13]]
                            chunk = np.tile(anchor, (50, 1))
                        else:
                            sampler.sampler.task = request.get("task", config.get("task", ""))
                            chunk = sampler.infer(obs, prev_action_chunk=request.get("prefix"),
                                                  prefix_start=int(request.get("prefix_start", 1)))
                        connection.send({"ok": True, "targets": chunk})
                    except Exception as exc:
                        connection.send({"ok": False, "error": str(exc)})
    except BaseException as exc:
        progress("failed", detail=str(exc))
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
