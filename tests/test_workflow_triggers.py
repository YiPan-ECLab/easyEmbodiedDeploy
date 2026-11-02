"""Joint sum conditions, bounded windows and physical-controller ownership."""

import copy
import tempfile
import time
import unittest

import numpy as np

from test_workflow_continuous import plan, Pool, wait_for
from easy_embodied_deploy.workflow.plan import validate
from easy_embodied_deploy.workflow.library import SharedPlanStore
from easy_embodied_deploy.workflow.runtime import WorkflowController
from easy_embodied_deploy.workflow.triggers import TakeoverSchedule


def joint_plan():
    p = plan()
    event = p['continuous_vla']['takeovers'][0]
    event['at_s'] = 1
    event['trigger'] = dict(type='right_joint_sum', end_s=3, sum_deg=30, reference='window', hold_s=.2)
    return p


def joints(j4=0, j6=0, j5=0):
    return [0, 0, 0, np.deg2rad(j4), np.deg2rad(j5), np.deg2rad(j6), .04]


class JointTriggerTests(unittest.TestCase):
    def schedule(self, p=None):
        return TakeoverSchedule(validate(p or joint_plan())['continuous_vla']['takeovers'])

    def test_no_feedback_outside_window_and_no_late_forced_takeover(self):
        s = self.schedule()
        def unexpected(): raise AssertionError('feedback outside window')
        self.assertIsNone(s.poll(.9, unexpected))
        self.assertIsNone(s.poll(3.01, unexpected))
        self.assertEqual(s.states['e']['state'], 'expired')
        self.assertEqual(len(s.messages), 1)
        self.assertIsNone(s.poll(4, unexpected))
        self.assertEqual(len(s.messages), 1)

    def test_sum_10_plus_20_matches_inclusively_and_fires_once(self):
        s = self.schedule()
        s.poll(1, lambda:joints())
        self.assertIsNone(s.poll(1.05, lambda:joints(10, 0)))
        self.assertIsNone(s.poll(1.1, lambda:joints(0, 20)))
        self.assertIsNone(s.poll(1.15, lambda:joints(10, 20)))
        self.assertIsNone(s.poll(1.25, lambda:joints(10, 20)))
        self.assertEqual(s.poll(1.35, lambda:joints(10, 20))['id'], 'e')
        self.assertIsNone(s.poll(1.45, lambda:joints(10, 20)))
        self.assertAlmostEqual(s.states['e']['triggered_at_s'], 1.35)

    def test_reverse_direction_and_one_frame_spike_never_trigger(self):
        s = self.schedule();s.poll(1, lambda:joints())
        s.poll(1.05, lambda:joints(-10, -20))
        s.poll(1.1, lambda:joints(10, 20))
        s.poll(1.15, lambda:joints())
        for t in [1.25, 1.35, 1.45]:
            self.assertIsNone(s.poll(t, lambda:joints()))
        self.assertEqual(s.states['e']['held_s'], 0)

    def test_drop_below_sum_threshold_restarts_dwell(self):
        s = self.schedule();s.poll(1, lambda:joints())
        s.poll(1.05, lambda:joints(10, 20))
        s.poll(1.15, lambda:joints(10, 19))
        s.poll(1.25, lambda:joints(10, 20))
        self.assertIsNone(s.poll(1.35, lambda:joints(10, 20)))
        self.assertIsNotNone(s.poll(1.45, lambda:joints(10, 20)))

    def test_pause_or_another_takeover_resets_dwell_keeps_reference(self):
        s = self.schedule();s.poll(1, lambda:joints(10, 20))
        s.poll(1.05, lambda:joints(20, 40));s.interrupt()
        s.poll(1.15, lambda:joints(20, 40))
        self.assertIsNone(s.poll(1.25, lambda:joints(20, 40)))
        self.assertIsNotNone(s.poll(1.35, lambda:joints(20, 40)))
        np.testing.assert_allclose(s.states['e']['baseline_deg'], [10, 20])

    def test_sampling_gap_cannot_count_as_continuous_confirmation(self):
        s = self.schedule();s.poll(1, lambda:joints())
        s.poll(1.05, lambda:joints(10, 20))
        self.assertIsNone(s.poll(1.5, lambda:joints(10, 20)))
        self.assertEqual(s.states['e']['held_s'], 0)

    def test_manual_reference_and_inclusive_window_end(self):
        p = joint_plan();t=p['continuous_vla']['takeovers'][0]['trigger']
        t.update(reference='manual', j4_reference_deg=20, j6_reference_deg=30)
        s = self.schedule(p)
        for now in [2.8, 2.9]:self.assertIsNone(s.poll(now, lambda:joints(30, 50)))
        self.assertIsNotNone(s.poll(3, lambda:joints(30, 50)))

    def test_unmatched_condition_does_not_block_later_timer(self):
        p=joint_plan();e=copy.deepcopy(p['continuous_vla']['takeovers'][0])
        e.update(id='timer',at_s=1.1,trigger={'type':'time'})
        p['continuous_vla']['takeovers'].append(e)
        s=self.schedule(p);s.poll(1,lambda:joints())
        self.assertEqual(s.poll(1.1,lambda:joints())['id'],'timer')
        self.assertNotEqual(s.states['e']['state'],'triggered')

    def test_invalid_feedback_raises_instead_of_triggering(self):
        for value in [[0]*6, joints(float('nan'),0), joints(float('inf'),0)]:
            with self.assertRaisesRegex(RuntimeError,'有效'):
                self.schedule().poll(1,lambda:value)

    def test_validation_rejects_invalid_or_ambiguous_conditions(self):
        for updates in [dict(type='unknown'),dict(end_s=1),dict(end_s=.5),dict(end_s=float('nan')),
                dict(sum_deg=0),dict(sum_deg=float('inf')),dict(sum_deg=-1),dict(sum_deg=721),
                dict(j4_reference_deg=float('nan'),reference='manual'),dict(hold_s=0),dict(hold_s=4),dict(reference='unknown'),
                dict(reference='manual',j6_reference_deg=None),dict(sum_deg=None)]:
            p=joint_plan();p['continuous_vla']['takeovers'][0]['trigger'].update(updates)
            with self.subTest(updates=updates),self.assertRaises(ValueError):validate(p)

    def test_conditions_roundtrip_and_old_timer_default(self):
        with tempfile.TemporaryDirectory() as directory:
            store=SharedPlanStore(directory);p=joint_plan();identifier=store.save(p)
            self.assertEqual(store.load(identifier)['continuous_vla'],validate(p)['continuous_vla'])
        self.assertEqual(validate(plan())['continuous_vla']['takeovers'][0]['trigger'],{'type':'time'})

    def test_j5_is_ignored_and_one_joint_may_supply_entire_sum(self):
        s=self.schedule();s.poll(1,lambda:joints())
        for t in [1.05,1.15,1.25]:
            self.assertIsNone(s.poll(t,lambda:joints(0,0,90)))
        s.poll(1.35,lambda:joints(0,30,-90))
        s.poll(1.45,lambda:joints(0,30,90))
        self.assertIsNotNone(s.poll(1.55,lambda:joints(0,30,-90)))
        self.assertAlmostEqual(s.states['e']['sum_deg'],30)

    def test_negative_angles_cancel_instead_of_absolute_sum(self):
        s=self.schedule();s.poll(1,lambda:joints())
        for t in [1.05,1.15,1.25]:
            self.assertIsNone(s.poll(t,lambda:joints(-20,40)))
        self.assertAlmostEqual(s.states['e']['sum_deg'],20)

    def test_old_j5_rule_migrates_to_sum_without_retaining_j5_fields(self):
        p=joint_plan();event=p['continuous_vla']['takeovers'][0]
        event['trigger']=dict(type='right_joints',end_s=3,hold_s=.2,reference='manual',
            j5_reference_deg=50,j6_reference_deg=10,j5_deg=2,j6_deg=30,
            j5_direction=-1,j6_direction=1)
        checked=validate(p)['continuous_vla']['takeovers'][0]['trigger']
        self.assertEqual(checked,dict(type='right_joint_sum',end_s=3,hold_s=.2,
            reference='manual',j4_reference_deg=0,j6_reference_deg=10,sum_deg=30))
        self.assertEqual(event['trigger']['type'],'right_joints')

    def test_real_controller_triggers_and_returns_to_fresh_background_inference(self):
        class WristPool(Pool):
            def infer(self, observation, task, prefix, prefix_start, cancel_event=None):
                q=np.array(observation['observation.state']);q[10]+=.03;q[12]+=.06
                self.calls.append((self.controller.status.get('controller_owner'),time.monotonic(),q.copy()))
                time.sleep(.003)
                return np.tile(q,(8,1))
        pool=WristPool();c=WorkflowController(pool);pool.controller=c
        c.connect(True);self.addCleanup(c.close)
        p=joint_plan();e=p['continuous_vla']['takeovers'][0];e['at_s']=0
        e['trigger'].update(end_s=4,sum_deg=3,hold_s=.05)
        c.start(p,home_before=False)
        wait_for(lambda:any('自动交还' in e['message'] for e in c.events))
        self.assertEqual(c.snapshot()['takeover_conditions'][0]['state'],'triggered')
        self.assertGreater(sum(owner=='points' for owner,_,_ in pool.calls),0)
        self.assertTrue(c.last_verification['reached']);self.assertTrue(c.busy())
        self.assertEqual(sum('关节条件关键点接管' in e['message'] for e in c.events),1)
        c.stop();wait_for(lambda:not c.busy())




class JointPoseTriggerTests(unittest.TestCase):
    def pose_plan(self):
        p=joint_plan();p['continuous_vla']['takeovers'][0]['trigger']=dict(
            type='right_j6_pose',end_s=3,j6_deg=-10,z_mm=500,hold_s=.2)
        return p

    def schedule(self):
        return TakeoverSchedule(validate(self.pose_plan())['continuous_vla']['takeovers'])

    def feedback(self, j6=0, raised=True):
        # Two kinematic test poses: high flange and low flange.
        return [0, .5 if raised else 2, -1.5, 0, 0, np.deg2rad(j6), .04]

    def test_requires_both_conditions_and_resets_on_either_drop(self):
        s=self.schedule()
        for t in [1,1.1,1.2]:self.assertIsNone(s.poll(t,lambda:self.feedback(-11)))
        for t in [1.3,1.4,1.5]:self.assertIsNone(s.poll(t,lambda:self.feedback(30,False)))
        s.poll(1.6,lambda:self.feedback())
        s.poll(1.7,lambda:self.feedback(-11))
        s.poll(1.8,lambda:self.feedback())
        self.assertIsNone(s.poll(1.9,lambda:self.feedback()))
        self.assertIsNotNone(s.poll(2,lambda:self.feedback()))
        self.assertIsNone(s.poll(2.1,lambda:self.feedback()))

    def test_inclusive_thresholds_and_status(self):
        from easy_embodied_deploy.workflow.triggers import right_flange_xyz
        p=self.pose_plan();t=p['continuous_vla']['takeovers'][0]['trigger'];q=self.feedback(-10)
        t['z_mm']=float(right_flange_xyz(q)[2]*1000)
        s=TakeoverSchedule(validate(p)['continuous_vla']['takeovers'])
        s.poll(1,lambda:q);s.poll(1.1,lambda:q)
        self.assertIsNotNone(s.poll(1.2,lambda:q))
        state=s.snapshot()[0];self.assertEqual(state['type'],'right_j6_pose')
        self.assertAlmostEqual(state['j6_deg'],-10);self.assertAlmostEqual(state['z_mm'],t['z_mm'])
        self.assertNotIn('sum_deg',state)

    def test_window_pause_and_sampling_gaps(self):
        s=self.schedule()
        def unexpected():raise AssertionError('read outside window')
        s.poll(.99,unexpected);s.poll(1,lambda:self.feedback());s.interrupt()
        s.poll(1.1,lambda:self.feedback());self.assertIsNone(s.poll(1.2,lambda:self.feedback()))
        self.assertIsNone(s.poll(1.6,lambda:self.feedback()))
        s.poll(3.01,unexpected);self.assertEqual(s.snapshot()[0]['state'],'expired')

    def test_invalid_config_and_feedback(self):
        for changes in [dict(j6_deg=121),dict(j6_deg=-121),dict(z_mm=751),dict(z_mm=float('nan')),
                        dict(j6_deg=None),dict(end_s=.5),dict(hold_s=0)]:
            p=self.pose_plan();p['continuous_vla']['takeovers'][0]['trigger'].update(changes)
            with self.subTest(changes=changes),self.assertRaises(ValueError):validate(p)
        for q in [[0]*6,self.feedback(float('nan'))]:
            with self.assertRaises(RuntimeError):self.schedule().poll(1,lambda:q)

    def test_roundtrip_discards_obsolete_sum_fields(self):
        p=self.pose_plan();p['continuous_vla']['takeovers'][0]['trigger'].update(sum_deg=30,reference='manual')
        checked=validate(p);self.assertNotIn('sum_deg',checked['continuous_vla']['takeovers'][0]['trigger'])
        with tempfile.TemporaryDirectory() as d:
            store=SharedPlanStore(d);identifier=store.save(p)
            self.assertEqual(store.load(identifier)['continuous_vla'],checked['continuous_vla'])

    def test_actual_runtime_takeover_and_return_with_mock_hardware(self):
        pool=Pool();c=WorkflowController(pool);pool.controller=c;c.connect(True);self.addCleanup(c.close)
        c.hardware.joints[8:10]=[.5,-1.5]
        p=self.pose_plan();e=p['continuous_vla']['takeovers'][0];e['at_s']=0;e['trigger'].update(end_s=4,z_mm=500,hold_s=.05)
        c.start(p,home_before=False)
        wait_for(lambda:any('自动交还' in e['message'] for e in c.events))
        self.assertEqual(c.snapshot()['takeover_conditions'][0]['state'],'triggered')
        self.assertTrue(c.busy());self.assertTrue(c.last_verification['reached'])
        self.assertGreater(sum(owner=='points' for owner,_,_ in pool.calls),0)
        c.stop();wait_for(lambda:not c.busy())


class JointSumPoseTests(unittest.TestCase):
    def config(self):
        p=joint_plan();p['continuous_vla']['takeovers'][0]['trigger']=dict(
            type='right_joint_sum_pose',end_s=3,sum_deg=30,z_mm=500,hold_s=.2)
        return p

    def schedule(self):
        return TakeoverSchedule(validate(self.config())['continuous_vla']['takeovers'])

    def feedback(self,j4=99,j6=-68,high=True):
        return [0,.5 if high else 2,-1.5,np.deg2rad(j4),0,np.deg2rad(j6),.04]

    def test_large_positive_j4_negative_j6_can_match_and_fires_once(self):
        s=self.schedule();s.poll(1,lambda:self.feedback());s.poll(1.1,lambda:self.feedback())
        self.assertIsNotNone(s.poll(1.2,lambda:self.feedback()))
        state=s.snapshot()[0];self.assertEqual(state['type'],'right_joint_sum_pose')
        self.assertAlmostEqual(state['sum_deg'],31);self.assertTrue(state['sum_matched'])
        self.assertTrue(state['z_matched']);self.assertNotIn('j6_threshold_deg',state)
        self.assertIsNone(s.poll(1.3,lambda:self.feedback()))

    def test_signed_sum_cancels_and_has_no_individual_joint_minimum(self):
        s=self.schedule()
        for t in [1,1.1,1.2]:self.assertIsNone(s.poll(t,lambda:self.feedback(-99,68)))
        self.assertAlmostEqual(s.snapshot()[0]['sum_deg'],-31)
        # Neither joint has its own minimum; J6 may supply the whole sum.
        s.poll(1.3,lambda:self.feedback(0,30));s.poll(1.4,lambda:self.feedback(0,30))
        self.assertIsNotNone(s.poll(1.5,lambda:self.feedback(0,30)))

    def test_both_angle_and_height_required_and_dwell_resets(self):
        s=self.schedule()
        for t in [1,1.1,1.2]:self.assertIsNone(s.poll(t,lambda:self.feedback(high=False)))
        s.poll(1.3,lambda:self.feedback());s.poll(1.4,lambda:self.feedback(j4=90))
        s.poll(1.5,lambda:self.feedback());self.assertIsNone(s.poll(1.6,lambda:self.feedback()))
        self.assertIsNotNone(s.poll(1.7,lambda:self.feedback()))

    def test_pause_and_feedback_gaps_reset_dwell(self):
        s=self.schedule();s.poll(1,lambda:self.feedback());s.interrupt()
        s.poll(1.1,lambda:self.feedback());self.assertIsNone(s.poll(1.2,lambda:self.feedback()))
        self.assertIsNone(s.poll(1.6,lambda:self.feedback()))
        self.assertEqual(s.snapshot()[0]['held_s'],0)

    def test_expiry_never_forces_takeover_or_reads_outside_window(self):
        s=self.schedule()
        def unexpected():raise AssertionError('read outside window')
        self.assertIsNone(s.poll(.9,unexpected));self.assertIsNone(s.poll(3.01,unexpected))
        self.assertEqual(s.snapshot()[0]['state'],'expired')

    def test_validation_and_roundtrip_remove_obsolete_fields(self):
        p=self.config();p['continuous_vla']['takeovers'][0]['trigger'].update(j6_deg=25,reference='manual')
        checked=validate(p);t=checked['continuous_vla']['takeovers'][0]['trigger']
        self.assertNotIn('j6_deg',t);self.assertNotIn('reference',t)
        with tempfile.TemporaryDirectory() as d:
            store=SharedPlanStore(d);key=store.save(p)
            self.assertEqual(store.load(key)['continuous_vla'],checked['continuous_vla'])
        for change in [dict(sum_deg=None),dict(sum_deg=float('nan')),dict(sum_deg=221),dict(sum_deg=-221),dict(z_mm=751)]:
            p=self.config();p['continuous_vla']['takeovers'][0]['trigger'].update(change)
            with self.assertRaises(ValueError):validate(p)

    def test_exact_sum_and_height_boundaries_are_inclusive(self):
        from easy_embodied_deploy.workflow.triggers import right_flange_xyz
        p=self.config();q=self.feedback(j4=98,j6=-68)
        p['continuous_vla']['takeovers'][0]['trigger']['z_mm']=float(right_flange_xyz(q)[2]*1000)
        s=TakeoverSchedule(validate(p)['continuous_vla']['takeovers'])
        s.poll(1,lambda:q);s.poll(1.1,lambda:q);self.assertIsNotNone(s.poll(1.2,lambda:q))

    def test_controller_records_new_rule_and_returns_to_vla(self):
        from unittest.mock import patch
        from pathlib import Path
        import json
        with tempfile.TemporaryDirectory() as d,patch('easy_embodied_deploy.module_http.ROOT',Path(d)):
            pool=Pool();c=WorkflowController(pool);pool.controller=c;c.connect(True);self.addCleanup(c.close)
            c.hardware.joints[8:10]=[.5,-1.5];c.hardware.joints[10]=np.deg2rad(99);c.hardware.joints[12]=np.deg2rad(-68)
            p=self.config();e=p['continuous_vla']['takeovers'][0];e['at_s']=0;e['trigger'].update(end_s=4,hold_s=.05)
            c.start(p,home_before=False);wait_for(lambda:any('自动交还' in e['message'] for e in c.events))
            c.stop();wait_for(lambda:not c.busy())
            self.assertEqual(c.snapshot()['takeover_conditions'][0]['state'],'triggered')
            folder=Path(c.snapshot()['motion_recording']['path'])
            events=[json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
            fired=next(e['data']['conditions'][0] for e in events if e['kind']=='condition' and e['data']['conditions'][0]['state']=='triggered')
            self.assertEqual(fired['sum_threshold_deg'],30);self.assertTrue(fired['sum_matched'])
            self.assertTrue(fired['z_matched']);self.assertNotIn('j6_threshold_deg',fired)

class HighZOverrideTests(unittest.TestCase):
    def config(self, high=500):
        p=joint_plan()
        p['continuous_vla']['takeovers'][0]['trigger']=dict(
            type='right_joint_sum_pose',end_s=3,sum_deg=27,z_mm=300,
            high_z_mm=high,hold_s=.2)
        return p

    def schedule(self, high=500, observe=None):
        return TakeoverSchedule(validate(self.config(high))['continuous_vla']['takeovers'],observe=observe)

    def q(self, total=0):
        return [0,.5,-1.5,np.deg2rad(total),0,0,.04]

    def poll_z(self, s, t, z, total=0):
        from unittest.mock import patch
        with patch('easy_embodied_deploy.workflow.triggers.right_flange_xyz',return_value=np.array([0,0,z/1000])):
            return s.poll(t,lambda:self.q(total))

    def test_high_z_alone_inclusive_and_one_shot_with_branch_status(self):
        s=self.schedule()
        self.assertIsNone(self.poll_z(s,1,500))
        self.assertIsNone(self.poll_z(s,1.1,500))
        self.assertIsNotNone(self.poll_z(s,1.2,500))
        state=s.snapshot()[0]
        self.assertFalse(state['sum_matched']);self.assertFalse(state['combined_matched'])
        self.assertTrue(state['high_z_matched']);self.assertEqual(state['matched_by'],'high_z')
        self.assertIsNone(self.poll_z(s,1.3,500))

    def test_original_branch_still_works_below_high_z(self):
        s=self.schedule()
        for t in [1,1.1]:self.assertIsNone(self.poll_z(s,t,300,27))
        self.assertIsNotNone(self.poll_z(s,1.2,300,27))
        self.assertEqual(s.snapshot()[0]['matched_by'],'sum_and_height')

    def test_neither_branch_and_disabled_branch_do_not_trigger(self):
        for high in [500,None]:
            s=self.schedule(high)
            for t in [1,1.1,1.2]:self.assertIsNone(self.poll_z(s,t,499 if high else 600))
            self.assertIsNone(s.snapshot()[0]['matched_by'])

    def test_pause_gap_and_drop_reset_high_z_dwell(self):
        s=self.schedule();self.poll_z(s,1,500);s.interrupt()
        self.poll_z(s,1.1,500);self.assertIsNone(self.poll_z(s,1.2,500))
        self.assertIsNone(self.poll_z(s,1.6,500));self.assertEqual(s.snapshot()[0]['held_s'],0)
        self.poll_z(s,1.7,499);self.poll_z(s,1.8,500)
        self.assertIsNone(self.poll_z(s,1.9,500));self.assertIsNotNone(self.poll_z(s,2,500))

    def test_high_z_cannot_bypass_window(self):
        s=self.schedule()
        def unexpected():raise AssertionError('feedback outside window')
        self.assertIsNone(s.poll(.9,unexpected));self.assertIsNone(s.poll(3.01,unexpected))
        self.assertEqual(s.snapshot()[0]['state'],'expired')

    def test_both_branches_and_continuous_or_dwell(self):
        s=self.schedule();self.poll_z(s,1,300,27)
        self.poll_z(s,1.1,500,0)
        self.assertIsNotNone(self.poll_z(s,1.2,500,27))
        self.assertEqual(s.snapshot()[0]['matched_by'],'both')

    def test_validation_and_disabled_roundtrip(self):
        for high in [300,299,float('nan'),float('inf'),751]:
            with self.subTest(high=high),self.assertRaises(ValueError):validate(self.config(high))
        with tempfile.TemporaryDirectory() as d:
            store=SharedPlanStore(d);key=store.save(self.config(None))
            self.assertIsNone(store.load(key)['continuous_vla']['takeovers'][0]['trigger']['high_z_mm'])
        p=self.config();del p['continuous_vla']['takeovers'][0]['trigger']['high_z_mm']
        self.assertNotIn('high_z_mm',validate(p)['continuous_vla']['takeovers'][0]['trigger'])

    def test_high_only_controller_records_reason_and_returns_to_vla(self):
        from unittest.mock import patch
        from pathlib import Path
        import json
        with tempfile.TemporaryDirectory() as d,patch('easy_embodied_deploy.module_http.ROOT',Path(d)):
            pool=Pool();c=WorkflowController(pool);pool.controller=c;c.connect(True);self.addCleanup(c.close)
            c.hardware.joints[8:10]=[.5,-1.5]
            c.hardware.joints[[10,12]]=0
            p=self.config();e=p['continuous_vla']['takeovers'][0];e['at_s']=0
            e['trigger'].update(end_s=4,hold_s=.05)
            c.start(p,home_before=False);wait_for(lambda:any('自动交还' in e['message'] for e in c.events))
            c.stop();wait_for(lambda:not c.busy())
            folder=Path(c.snapshot()['motion_recording']['path'])
            events=[json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
            fired=next(e['data']['conditions'][0] for e in events if e['kind']=='condition' and e['data']['conditions'][0]['state']=='triggered')
            self.assertFalse(fired['sum_matched']);self.assertEqual(fired['matched_by'],'high_z')
            self.assertTrue(fired['high_z_matched']);self.assertEqual(fired['high_z_threshold_mm'],500)

if __name__=='__main__':unittest.main()
