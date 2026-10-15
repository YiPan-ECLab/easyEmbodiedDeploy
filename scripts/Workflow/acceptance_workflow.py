#!/usr/bin/env python3
"""Explicit small-motion follower acceptance; optional real inference never actuates."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from easy_embodied_deploy.workflow.runtime import WorkflowController
from easy_embodied_deploy.workflow.plan import new_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='authorize actual small follower motion')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-config', type=Path, help='also load and infer once, without executing policy output')
    args = parser.parse_args()
    if not args.execute:
        parser.error('--execute is required; this script moves the followers')
    if args.output.exists():
        parser.error('output already exists')
    args.output.mkdir(parents=True)
    report = dict(status='running', rounds=[], maximum_joint_deg=0, policy_actions_executed=False)
    controller = WorkflowController()
    initial_ok = False

    def check_pose():
        snapshot = controller.hardware.snapshot()
        joints = np.asarray(snapshot['joints'])
        maximum = float(np.max(np.abs(np.r_[joints[:6], joints[7:13]])))
        report['maximum_joint_deg'] = max(report['maximum_joint_deg'], float(np.degrees(maximum)))
        if maximum > np.deg2rad(6):
            controller.stop()
            raise RuntimeError('acceptance motion exceeded 6 degrees; stopped')
        return joints

    def wait_motion():
        deadline = time.monotonic()+30
        while controller.busy():
            check_pose()
            if time.monotonic() >= deadline:
                controller.stop()
                raise TimeoutError('motion did not finish')
            time.sleep(.02)
        if controller.status['state'] != 'completed':
            raise RuntimeError(controller.status.get('error') or controller.status['state'])

    try:
        controller.connect(False, {'depth_roles': []})
        initial = check_pose()
        if np.max(np.abs(np.r_[initial[:6], initial[7:13]])) > np.deg2rad(3):
            raise RuntimeError('start must be within 3 degrees of zero; no homing attempted')
        initial_ok = True
        plan = new_plan()
        plan['settings'].update(control_hz=50, joint_speed=.15, joint_acceleration=.3)
        if args.model_config:
            config = json.loads(args.model_config.read_text())
            plan['models'] = [config]
            controller.preload(plan, [config['id']])

        for command in ('move_js', 'move_j'):
            target = np.zeros(14)
            target[[6, 13]] = np.clip(initial[[6, 13]], 0, .1)
            target[0], target[7] = np.deg2rad(2), np.deg2rad(-2)
            target[8], target[9] = np.deg2rad(2), np.deg2rad(-2)
            plan['settings']['joint_command'] = command
            plan['points'] = [dict(id='target', name='小幅验收点', space='joint',
                                   arms=dict(left=target[:7].tolist(), right=target[7:].tolist()))]
            plan['steps'] = [dict(id='move', type='point', point='target', name=command, duration=1.2)]
            controller.start(plan, home_before=False)
            wait_motion()
            controller.teach(True, 'both')
            point = controller.capture('验收读点（不保存）', 'both')
            # No disconnect between recording and home, or between rounds.
            controller.home()
            wait_motion()
            actual = check_pose()
            zero_error = float(np.degrees(np.max(np.abs(np.r_[actual[:6], actual[7:13]]))))
            if zero_error > 1:
                raise AssertionError(f'zero error: {zero_error} deg')
            report['rounds'].append(dict(command=command, recorded=point['arms'], zero_error_deg=zero_error))
            print(command, 'capture -> home PASS', flush=True)

        if args.model_config:
            model = controller.models.get(config['id'])
            deadline = time.monotonic()+180
            while model.stage != 'ready':
                if model.stage == 'failed':
                    raise RuntimeError(model.detail)
                if time.monotonic() >= deadline:
                    raise TimeoutError('model loading exceeded 180s')
                time.sleep(.2)
            deadline = time.monotonic()+15
            while True:
                try:
                    observation = controller.hardware.observation()
                    break
                except RuntimeError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(.1)
            controller.preload(plan, [config['id']])
            if controller.models.get(config['id']) is not model:
                raise AssertionError('unchanged config unexpectedly reloaded model')
            started = time.monotonic()
            targets = model.infer(observation, config.get('task', ''))
            report['inference'] = dict(backend=config['backend'], action_mode=config['action_mode'],
                                      shape=list(targets.shape), seconds=time.monotonic()-started,
                                      cameras=[key for key in observation if key.startswith('observation.images.')],
                                      unchanged_model_reused=True)
            print('real camera/model inference PASS; output was NOT executed', flush=True)
        report['status'] = 'passed'
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        try:
            controller.stop()
            if controller.thread:
                controller.thread.join(5)
            if initial_ok and controller.hardware and not controller.hardware.emergency:
                # Only return from the deliberately small acceptance envelope.
                check_pose()
                controller.home()
                wait_motion()
                report['final_zero'] = check_pose().tolist()
        except Exception as error:
            report.update(status='failed', cleanup_error=str(error))
        finally:
            try:
                controller.close()
            except Exception as error:
                report.update(status='failed', close_error=str(error))
            finally:
                (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
