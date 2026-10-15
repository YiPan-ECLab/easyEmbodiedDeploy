#!/usr/bin/env python3
"""Explicit small-motion hardware acceptance, using the real recorder and cameras.

Requires --execute. Four arms must start within the recorder's zero tolerance. The test
uses a 2-degree J1 start, then a two-degree task motion on one side at a time.
A second SDK connection commands the physical leader; the recorder receives its
real CAN commands through the unchanged follow loop. No observations are mocked.
"""
import argparse
import json
import math
import sys
import threading
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pyAgxArm
import record
from easy_embodied_deploy.resources import hardware_lease
from easy_embodied_deploy.recording_start import normalize
from piper_recorder.robot import DualPiper, PiperPair
from piper_recorder.dataset import write_episode


def make_robot(start):
    return DualPiper(record.PIPERS,start_point=start,firmware=record.FIRMWARE,
        home_speed=record.HOME_SPEED,home_timeout_s=record.HOME_TIMEOUT_S,
        home_tolerance_deg=record.HOME_TOLERANCE_DEG,
        preposition_speed=record.PREPOSITION_SPEED,preposition_timeout_s=record.PREPOSITION_TIMEOUT_S,
        zero_tolerance_deg=record.ZERO_TOLERANCE_DEG,follow_hz=record.FOLLOW_HZ,
        status_period_s=record.STATUS_PERIOD_S,gripper_force=record.GRIPPER_FORCE_N,
        gripper_max_width=record.GRIPPER_MAX_WIDTH_M,
        gripper_home_timeout_s=record.GRIPPER_HOME_TIMEOUT_S,
        gripper_home_tolerance_m=record.GRIPPER_HOME_TOLERANCE_M)


class Envelope:
    def __init__(self):
        self.arms={};self.maximum={};self.error=None;self.stop=threading.Event()

    def check(self):
        if self.error:raise RuntimeError(self.error)

    def add(self, name, arm, gripper):
        self.arms[name]=arm;self.maximum[name]=0
        for method in ('move_j','move_js'):
            original=getattr(arm,method)
            def guarded(q,call=original,**kwargs):
                self.check()
                if len(q)!=6 or not np.isfinite(q).all() or np.max(np.abs(q))>math.radians(4.1):
                    raise RuntimeError(f'Acceptance rejected target outside 4.1 degrees: {q}')
                return call(q,**kwargs)
            setattr(arm,method,guarded)
        original=gripper.move_gripper_m
        def guarded_grip(width,*args,**kwargs):
            self.check()
            if not 0<=width<=.006:raise RuntimeError('Acceptance gripper target exceeds 6 mm')
            return original(width,*args,**kwargs)
        gripper.move_gripper_m=guarded_grip

    def monitor(self):
        while not self.stop.wait(.01):
            for name,arm in self.arms.items():
                msg=arm.get_joint_angles()
                if msg is None or time.time()-msg.timestamp>.2:continue
                amplitude=float(np.max(np.abs(np.degrees(msg.msg))))
                self.maximum[name]=max(self.maximum[name],amplitude)
                if amplitude>6:self.error=f'{name} actual excursion exceeded 6 degrees'


def verify_home(rig,label):
    result={}
    controlled={side:pair._leader_is_follower for side,pair in rig.pairs.items()}
    try:
        for side,pair in rig.pairs.items():
            pair._enter_leader_control_mode()
        time.sleep(.15)
        for side,pair in rig.pairs.items():
            for role in ('leader','follower'):
                arm=getattr(pair,role);grip=getattr(pair,role+'_gripper')
                msg=arm.get_joint_angles();g=grip.get_gripper_status()
                if msg is None or time.time()-msg.timestamp>.2 or g is None:
                    raise RuntimeError(f'{side}/{role}: feedback not fresh')
                q=np.array(msg.msg)
                error=float(np.max(np.abs(np.degrees(q-pair.home_joints))))
                # Use the recorder's physical-width normalization, while keeping
                # the raw encoder reading in the report (closed may read -0.2 mm).
                width_error=abs(max(0,min(record.GRIPPER_MAX_WIDTH_M,g.msg.value))-pair.home_width)
                if error>record.HOME_TOLERANCE_DEG or width_error>record.GRIPPER_HOME_TOLERANCE_M:
                    raise RuntimeError(f'{side}/{role}: home error {error} degrees, {width_error} m')
                result[side+'_'+role]=dict(joints_deg=np.degrees(q).tolist(),
                    joint_error_deg=error,gripper_m=g.msg.value,gripper_error_m=width_error)
    finally:
        for side,pair in rig.pairs.items():
            if not controlled[side]:pair._restore_teaching()
    print(label,json.dumps(result,ensure_ascii=False),flush=True)
    return result


class ScriptedOperator:
    def __init__(self,rig,side,actuator,envelope):
        self.rig,self.side,self.actuator,self.envelope=rig,side,actuator,envelope
        self.started=None

    def phase(self,phase,**kwargs):
        self.envelope.check()
        if phase=='waiting_motion':
            self.rig.pairs[self.side]._enter_leader_control_mode()
            self.started=time.monotonic()

    def update(self,*args,**kwargs):self.envelope.check()

    def stop_requested(self):
        self.envelope.check()
        elapsed=time.monotonic()-self.started
        if elapsed>=3:return True
        fraction=min(1,max(0,(elapsed-.2)/2))
        q=list(self.rig.pairs[self.side].home_joints)
        q[0]+=(1 if self.side=='left' else -1)*math.radians(2)*(3*fraction**2-2*fraction**3)
        self.actuator.move_js(q)
        return False


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if not args.execute:parser.error('Real motion requires --execute')
    args.output.mkdir(parents=True,exist_ok=False)
    report={'status':'running','episodes':[],'checks':{},'motion_limit_deg':6}
    def save(): (args.output/'hardware_acceptance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    save()
    envelope=Envelope();rig=None;preflight=False
    with hardware_lease(),ExitStack() as resources:
        try:
            cameras=record.open_cameras(record.CAMERA_SERIALS,resources,width=record.CAMERA_WIDTH,
                height=record.CAMERA_HEIGHT,fps=record.CAMERA_FPS,image_size=record.IMAGE_SIZE,
                max_age_s=record.CAMERA_MAX_AGE_S)
            start=normalize(dict(mode='custom',name='小幅验收起点',arms={
                side:[math.radians(sign*2),0,0,0,0,0,.005]
                for side,sign in [('left',1),('right',-1)]}))
            report['start_point']=start
            rig=make_robot(start);resources.callback(rig.close);rig.connect()
            for pair in rig.pairs.values():pair._enter_leader_control_mode()
            time.sleep(.2)
            for side,pair in rig.pairs.items():
                for role in ('leader','follower'):
                    arm=getattr(pair,role);grip=getattr(pair,role+'_gripper')
                    msg=arm.get_joint_angles();status=arm.get_arm_status()
                    if (msg is None or time.time()-msg.timestamp>.2 or np.max(np.abs(np.degrees(msg.msg)))>record.ZERO_TOLERANCE_DEG
                            or status is None or int(status.msg.arm_status)!=0):
                        raise RuntimeError(f'{side}/{role}: not healthy and within recorder zero tolerance')
                    for i in range(1,7):
                        driver=arm.get_driver_states(i)
                        deadline=time.monotonic()+3
                        while driver is None and time.monotonic()<deadline:
                            time.sleep(.02);driver=arm.get_driver_states(i)
                        if driver is None:
                            raise RuntimeError(f'{side}/{role}/J{i}: missing driver feedback')
                        if int(driver.msg.foc_status_code)&~0x40:
                            raise RuntimeError(f'{side}/{role}/J{i}: driver fault 0x{driver.msg.foc_status_code:02X}')
                    envelope.add(side+'_'+role,arm,grip)
            preflight=True
            # Independent SocketCAN receiver: no synthetic state/action injection.
            actuators={}
            for side,pair in rig.pairs.items():
                arm=pyAgxArm.AgxArmFactory.create_arm(pyAgxArm.create_agx_arm_config(
                    robot=pyAgxArm.ArmModel.PIPER,firmeware_version=record.FIRMWARE,
                    channel=pair.leader_name,local_loopback=True))
                grip=arm.init_effector(arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
                resources.callback(arm.disconnect);arm.connect()
                actuators[side]=arm;envelope.add(side+'_actuator',arm,grip)
            monitor=threading.Thread(target=envelope.monitor,daemon=True);monitor.start()
            rig.home();report['checks']['initial_home']=verify_home(rig,'INITIAL_HOME');save()
            for side in ('left','right'):
                operator=ScriptedOperator(rig,side,actuators[side],envelope)
                episode=record.record_episode(rig,cameras,record.RECORD_HZ,operator=operator)
                report['checks'][side+'_return']=verify_home(rig,side.upper()+'_RETURN')
                path=write_episode(episode,f'Acceptance: move only the {side} arm by two degrees',
                    args.output/side,pyagxarm_version=getattr(pyAgxArm,'__version__','unknown'),
                    firmware_profile=str(record.FIRMWARE),firmware_version=record.FIRMWARE_VERSION,
                    gripper_force=record.GRIPPER_FORCE_N,record_hz=record.RECORD_HZ)
                report['episodes'].append(str(path.resolve()));save()
            envelope.check();report['status']='passed'
        except BaseException as exc:
            report.update(status='failed',error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            if rig and preflight:
                try:
                    for pair in rig.pairs.values():
                        pair.custom_start=False;pair.home_joints=(0.,)*6;pair.home_width=0.
                    rig.home();report['checks']['restored_zero']=verify_home(rig,'RESTORED_ZERO')
                except Exception as exc:
                    report.update(status='failed',restore_error=str(exc))
            elif rig:
                for pair in rig.pairs.values():
                    if pair.leader is not None:pair._restore_teaching()
            envelope.stop.set()
            if 'monitor' in locals():monitor.join(2)
            report['maximum_absolute_joint_deg']=envelope.maximum
            save()
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
