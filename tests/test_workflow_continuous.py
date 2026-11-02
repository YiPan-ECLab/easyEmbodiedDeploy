"""Offline ownership, inference freshness, schedule and cancellation contracts."""

import copy
import sys
import tempfile
import time
import unittest
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from easy_embodied_deploy.workflow.continuous import ContinuousPolicy
from easy_embodied_deploy.workflow.library import SharedPlanStore
from easy_embodied_deploy.workflow.plan import new_plan, validate
from easy_embodied_deploy.workflow.runtime import WorkflowController


def plan():
    p = new_plan()
    p['mode'] = 'continuous_vla'
    p['points'] = [dict(id='p', name='接管点', space='joint', arms={'right':[.02,.2,-.2,0,0,.785,.04]})]
    p['models'] = [dict(id='m',name='持续模型',backend='mock',action_mode='absolute_joint',policy_hz=30,control_hz=100,rtc_queue_threshold=3)]
    p['continuous_vla'] = dict(model='m',takeovers=[dict(id='e',name='定时段',at_s=.1,
        steps=[dict(id='s',type='point',point='p',duration=.5,blend=True)])])
    return p


def wait_for(predicate, timeout=5):
    deadline = time.monotonic()+timeout
    while not predicate():
        if time.monotonic()>deadline:
            raise AssertionError('timed out')
        time.sleep(.005)


class Pool:
    def __init__(self):
        self.model = None
        self.calls = []
        self.controller = None
        self.delay = .003

    def preload(self, configs):
        self.model = SimpleNamespace(config=copy.deepcopy(configs[0]),stage='ready',contract=None,
                                     inference={},latencies=[self.delay],infer=self.infer)

    def infer(self, observation, task, prefix, prefix_start, cancel_event=None):
        self.calls.append((self.controller.status.get('controller_owner'),time.monotonic(),np.array(observation['observation.state']).copy()))
        time.sleep(self.delay)
        q = np.array(observation['observation.state'])
        q[7] += .01
        return np.tile(q,(8,1))

    def get(self, identifier): return self.model
    def snapshot(self): return []
    def close(self, identifier=None): pass


class ContinuousTests(unittest.TestCase):
    def controller(self):
        pool=Pool()
        c=WorkflowController(pool)
        pool.controller=c
        c.connect(True)
        self.addCleanup(c.close)
        return c,pool

    def test_background_inference_continues_during_points_and_returns_without_ending(self):
        c,pool=self.controller()
        original=plan()
        before=copy.deepcopy(original)
        c.start(original,home_before=False)
        wait_for(lambda:any('自动交还' in e['message'] for e in c.events))
        wait_for(lambda:c.status.get('controller_owner')=='vla' and c.status['actions']>3)
        self.assertTrue(c.busy())
        self.assertEqual(original,before)
        self.assertGreater(sum(owner=='points' for owner,_,_ in pool.calls),3)
        self.assertTrue(c.last_verification['reached'])
        self.assertFalse(c.snapshot()['error'])
        c.stop();wait_for(lambda:not c.busy())
        self.assertEqual(c.status['state'],'stopped')

    def test_pause_freezes_schedule_and_sends_no_commands_then_resumes(self):
        c,pool=self.controller()
        p=plan();p['continuous_vla']['takeovers'][0]['at_s']=.6
        c.start(p,home_before=False)
        wait_for(lambda:c.status['actions']>=2)
        c.pause();time.sleep(.08)
        elapsed=c.continuous_elapsed;commands=len(c.hardware.commands);calls=len(pool.calls)
        time.sleep(.15)
        self.assertEqual(len(c.hardware.commands),commands)
        self.assertEqual(len(pool.calls),calls)
        self.assertAlmostEqual(c.continuous_elapsed,elapsed,places=3)
        c.resume()
        wait_for(lambda:any('自动交还' in e['message'] for e in c.events))
        c.stop();wait_for(lambda:not c.busy())

    def test_zero_time_group_and_overdue_group_run_serially(self):
        c,_=self.controller()
        p=plan();first=p['continuous_vla']['takeovers'][0];first['at_s']=0
        second=copy.deepcopy(first);second.update(id='second',at_s=.05,name='第二段')
        p['continuous_vla']['takeovers'].append(second)
        c.start(p,home_before=False)
        wait_for(lambda:sum('自动交还' in e['message'] for e in c.events)==2)
        starts=[e['message'] for e in c.events if e['message'].startswith('定时关键点接管')]
        self.assertEqual(starts,['定时关键点接管：定时段','定时关键点接管：第二段'])
        c.stop();wait_for(lambda:not c.busy())

    def test_failed_inference_stops_and_holds(self):
        c,pool=self.controller()
        def fail(*args,**kwargs): raise RuntimeError('inference unavailable')
        pool.infer=fail
        c.start(plan(),home_before=False)
        wait_for(lambda:not c.busy())
        self.assertEqual(c.status['state'],'failed')
        self.assertIn('inference unavailable',c.status['error'])

    def test_emergency_during_point_takeover_is_latched(self):
        c,_=self.controller()
        c.start(plan(),home_before=False)
        wait_for(lambda:c.status.get('controller_owner')=='points')
        c.stop(emergency=True);wait_for(lambda:not c.busy())
        self.assertEqual(c.status['state'],'emergency')
        self.assertTrue(c.hardware.emergency)

    def test_stale_pending_chunk_is_discarded_after_ownership_switch(self):
        pending=Future();fresh=Future()
        model=SimpleNamespace(config={'policy_hz':10},stage='ready',latencies=[],inference={})
        controller=SimpleNamespace(models=SimpleNamespace(get=lambda _:model),_submit_model=lambda *args:fresh)
        policy=ContinuousPolicy(controller,{'model':'m'})
        policy.pending=pending
        policy.suppress(True);policy.suppress(False)
        pending.set_result(np.full((3,14),99.))
        policy.tick()
        self.assertIsNone(policy.desired)
        self.assertIs(policy.pending,fresh)
        fresh.set_result(np.zeros((3,14)))
        policy.tick()
        np.testing.assert_array_equal(policy.desired,np.zeros(14))

    def test_invalid_schedule_rejected_and_legacy_plan_unchanged(self):
        p=plan()
        for value in [-1,float('nan'),float('inf')]:
            p['continuous_vla']['takeovers'][0]['at_s']=value
            with self.assertRaises(ValueError):validate(p)
        p=plan();p['continuous_vla']['takeovers'][0]['steps'][0]['type']='vla'
        with self.assertRaises(ValueError):validate(p)
        p=plan();p['continuous_vla']['takeovers'][0]['steps'][0]['point']='missing'
        with self.assertRaises(ValueError):validate(p)
        p=plan();p['continuous_vla']['takeovers'].append(copy.deepcopy(p['continuous_vla']['takeovers'][0]))
        with self.assertRaises(ValueError):validate(p)
        old=new_plan();old.pop('mode');old.pop('continuous_vla')
        self.assertEqual(validate(old)['mode'],'sequential')

    def test_shared_save_reload_and_resource_deletion_prunes_schedule(self):
        with tempfile.TemporaryDirectory() as directory:
            store=SharedPlanStore(directory)
            identifier=store.save(plan())
            loaded=store.load(identifier)
            self.assertEqual(loaded['continuous_vla'],validate(plan())['continuous_vla'])
            store.delete_resource('points','p')
            self.assertEqual(store.load(identifier)['continuous_vla']['takeovers'],[])
            store.delete_resource('models','m')
            self.assertEqual(store.load(identifier)['mode'],'sequential')


if __name__=='__main__': unittest.main()
