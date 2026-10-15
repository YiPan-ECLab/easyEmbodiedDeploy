import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import record
from piper_recorder.robot import DualPiper, PiperPair
from easy_embodied_deploy import recording_start
from easy_embodied_deploy.recorder_console import command_for, RecorderSession


def robot(start=None):
    return DualPiper(record.PIPERS, start_point=start, firmware=record.FIRMWARE,
        home_speed=record.HOME_SPEED,home_timeout_s=record.HOME_TIMEOUT_S,
        home_tolerance_deg=record.HOME_TOLERANCE_DEG,
        preposition_speed=record.PREPOSITION_SPEED,preposition_timeout_s=record.PREPOSITION_TIMEOUT_S,
        zero_tolerance_deg=record.ZERO_TOLERANCE_DEG,follow_hz=record.FOLLOW_HZ,
        status_period_s=record.STATUS_PERIOD_S,gripper_force=record.GRIPPER_FORCE_N,
        gripper_max_width=record.GRIPPER_MAX_WIDTH_M,
        gripper_home_timeout_s=record.GRIPPER_HOME_TIMEOUT_S,
        gripper_home_tolerance_m=record.GRIPPER_HOME_TOLERANCE_M)


class StartPointTests(unittest.TestCase):
    def target(self):
        return dict(mode='custom',name='起点',arms=dict(
            left=[.1,.2,-.3,0,0,-.4,.025],right=[-.1,.3,-.4,0,0,.4,.04]))

    def test_custom_targets_reach_both_leaders_followers_and_grippers(self):
        self.check_home(self.target())

    def test_zero_keeps_leader_planning_and_follower_tracking(self):
        self.check_home(None)

    def test_custom_return_after_recording_keeps_leader_follower_trajectory(self):
        self.check_home(self.target(),following=True)

    def check_home(self, start, following=False):
        rig=robot(start);self.addCleanup(rig.close)
        calls=[]
        class Arm:
            def __init__(self,name):self.name=name;self.q=[0]*6
            def set_speed_percent(self,value):pass
            def enable(self):return True
            def move_j(self,q):self.q=list(q);calls.append((self.name,'j',list(q)))
            def move_js(self,q):self.q=list(q);calls.append((self.name,'js',list(q)))
            def get_joint_angles(self):return SimpleNamespace(msg=self.q,timestamp=time.time())
            def get_arm_status(self):return SimpleNamespace(msg=SimpleNamespace(arm_status=0))
        class Gripper:
            def __init__(self,name):self.name=name;self.width=0
            def move_gripper_m(self,width,force):self.width=width;calls.append((self.name,'grip',width))
            def get_gripper_status(self):return SimpleNamespace(timestamp=time.time(),msg=SimpleNamespace(mode='width',value=self.width))
        for side,pair in rig.pairs.items():
            pair.leader=Arm(side+'-leader');pair.follower=Arm(side+'-follower')
            pair.leader_gripper=Gripper(side+'-leader');pair.follower_gripper=Gripper(side+'-follower')
            pair._enter_leader_control_mode=Mock();pair._restore_teaching=Mock()
            if following:
                pair._thread=Mock()
                pair._thread.is_alive.side_effect=[True,False]
        rig.home()
        for side,pair in rig.pairs.items():
            q=start['arms'][side] if start else [0]*7
            self.assertEqual(pair.leader.q,q[:6]);self.assertEqual(pair.follower.q,q[:6])
            self.assertEqual(pair.leader_gripper.width,q[6]);self.assertEqual(pair.follower_gripper.width,q[6])
            self.assertEqual(list(pair._command.joints),q[:6])
            self.assertEqual(pair._restore_teaching.call_count,0 if start else 1)
        self.assertEqual(any(kind=='js' for _,kind,_ in calls),start is None or following)

    def test_capture_only_reads_selected_follower_and_rejects_stale_groups(self):
        from contextlib import nullcontext
        arm=Mock();arm._parser=SimpleNamespace()
        q=self.target()['arms']['right']
        for group in range(3):
            setattr(arm._parser,('joint_12','joint_34','joint_56')[group],
                    SimpleNamespace(timestamp=time.time(),msg=SimpleNamespace(**{
                        f'joint_{i+1}':q[i] for i in (group*2,group*2+1)})))
        arm.init_effector.return_value.get_gripper_status.side_effect=lambda:SimpleNamespace(
            timestamp=time.time(),msg=SimpleNamespace(mode='width',value=q[6]))
        with patch('easy_embodied_deploy.resources.hardware_lease',return_value=nullcontext()), \
             patch('pyAgxArm.AgxArmFactory.create_arm',return_value=arm) as factory:
            self.assertEqual(recording_start.capture('right'),{'right':q})
            self.assertEqual(factory.call_count,1)
            self.assertEqual({c[0] for c in arm.method_calls},
                             {'init_effector','connect','disconnect'})
            arm.init_effector.return_value.get_gripper_status.assert_called_once()
            arm._parser.joint_12.timestamp-=10
            with patch.object(recording_start.time,'monotonic',side_effect=[0,4]):
                with self.assertRaisesRegex(RuntimeError,'没有完整新鲜反馈'):
                    recording_start.capture('right')
            self.assertEqual(arm.disconnect.call_count,2)

    def test_zero_retains_preposition_custom_skips_it_and_homes_before_follow(self):
        for start in (None,self.target()):
            rig=robot(start);self.addCleanup(rig.close)
            order=[]
            rig.home=Mock(side_effect=lambda:order.append('home'))
            with patch.object(PiperPair,'preposition',lambda p:order.append('preposition')), \
                 patch.object(PiperPair,'start_following',lambda p:order.append('follow')):
                rig.preposition_and_start()
            self.assertEqual(order,['home','follow','follow'] if start else ['preposition','preposition','follow','follow'])

    def test_saved_start_roundtrip_and_cli_snapshot(self):
        with tempfile.TemporaryDirectory() as d, patch.object(recording_start,'START_PATH',Path(d)/'start.json'):
            self.assertEqual(recording_start.load()['mode'],'zero')
            recording_start.save(self.target())
            command,config=command_for({})
            self.assertEqual(json.loads(command[command.index('--start-point-json')+1]),self.target())
            self.assertEqual(config['start_point'],self.target())
            recording_start.save({'mode':'zero'})
            self.assertNotIn('--start-point-json',command_for({})[0])

    def test_setup_does_not_interrupt_active_recording(self):
        session=RecorderSession();session.process=Mock();session.process.poll.return_value=None
        with patch.object(recording_start,'capture') as capture:
            with self.assertRaises(RuntimeError):session.start_point_action('capture',{'side':'right'})
            capture.assert_not_called()

    def test_workflow_import_keeps_single_side_and_pose_feedback(self):
        with tempfile.TemporaryDirectory() as d,patch.object(recording_start,'ROOT',Path(d)):
            p=Path(d)/'datasets/workflows';p.mkdir(parents=True)
            (p/'test.json').write_text(json.dumps(dict(name='流程',points=[
                dict(name='右侧',space='joint',arms={'right':self.target()['arms']['right']}),
                dict(name='末端录点',space='pose',arms={'left':[0]*6+[.02]},
                     source={'feedback':{'left':{'joint_angles':[0,.2,-.2,0,0,0]}}})])))
            points=recording_start.workflow_points()
            self.assertEqual(set(points[0]['arms']),{'right'})
            self.assertEqual(points[1]['arms']['left'],[0,.2,-.2,0,0,0,.02])


if __name__=='__main__':unittest.main()
