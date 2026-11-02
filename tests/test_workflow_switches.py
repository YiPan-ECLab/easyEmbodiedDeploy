"""Timed multi-model switches, including switches during point ownership."""

import copy
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from easy_embodied_deploy.workflow.plan import validate, used_model_ids
from easy_embodied_deploy.workflow.library import SharedPlanStore
from easy_embodied_deploy.workflow.runtime import WorkflowController
from test_workflow_continuous import plan, wait_for


class MultiPool:
    def __init__(self):
        self.models={}
        self.calls=[]
        self.controller=None
        self.fail_late=False
        self.counts={}
        self.loads=0

    def preload(self, configs):
        self.loads+=1
        for config in configs:
            identifier=config['id']
            self.models[identifier]=SimpleNamespace(config=copy.deepcopy(config),stage='ready',detail='',
                contract=None,inference={},latencies=[.003],infer=lambda *a,_id=identifier,**kw:self.infer(_id,*a,**kw))

    def infer(self, identifier, observation, task, prefix, prefix_start, cancel_event=None):
        self.counts[identifier]=self.counts.get(identifier,0)+1
        self.calls.append((identifier,self.controller.status.get('controller_owner'),time.monotonic(),task))
        if self.fail_late and identifier=='m' and self.counts[identifier]>1:
            time.sleep(.2)
            raise RuntimeError('obsolete model request failed')
        time.sleep(.003)
        target=np.array(observation['command_anchor']).copy()
        target[7]+=.003 if identifier=='m' else -.003
        return np.tile(target,(8,1))

    def get(self, identifier):return self.models[identifier]
    def snapshot(self):return []
    def close(self, identifier=None):pass


def multi_plan():
    p=plan()
    p['continuous_vla']['takeovers']=[]
    other=copy.deepcopy(p['models'][0]);other.update(id='n',name='第二模型',control_hz=80)
    p['models'].append(other)
    p['continuous_vla']['switches']=[dict(id='switch1',at_s=.1,model='n',task='second task')]
    return p


class SwitchTests(unittest.TestCase):
    def controller(self):
        pool=MultiPool();c=WorkflowController(pool);pool.controller=c
        c.connect(True);self.addCleanup(c.close)
        return c,pool

    def test_a_b_a_reuses_preloaded_models_and_uses_each_configuration(self):
        c,pool=self.controller();p=multi_plan()
        p['continuous_vla']['switches'].append(dict(id='switch2',at_s=.35,model='m'))
        c.start(p,home_before=False)
        wait_for(lambda:sum('定时 VLA 切换' in e['message'] for e in c.events)==2)
        wait_for(lambda:c.tick_period==1/p['models'][0]['control_hz'] and not c.continuous_policy.suppressed)
        last=time.monotonic()
        wait_for(lambda:any(i=='m' and t>last for i,_,t,_ in pool.calls))
        self.assertEqual(set(pool.models),{'m','n'});self.assertEqual(pool.loads,1)
        self.assertTrue(any(i=='n' and task=='second task' for i,_,_,task in pool.calls))
        self.assertEqual(c.status['active_vla_id'],'m')
        self.assertEqual(c.tick_period,1/p['models'][0]['control_hz'])
        c.stop();wait_for(lambda:not c.busy())

    def test_switch_during_points_does_not_interrupt_group_and_returns_to_new_model(self):
        c,pool=self.controller();p=multi_plan()
        p['continuous_vla']['takeovers']=plan()['continuous_vla']['takeovers']
        p['continuous_vla']['takeovers'][0]['at_s']=.05
        p['continuous_vla']['switches'][0]['at_s']=.2
        c.start(p,home_before=False)
        wait_for(lambda:any('自动交还' in e['message'] for e in c.events))
        wait_for(lambda:any(i=='n' and owner=='vla' for i,owner,_,_ in pool.calls))
        self.assertGreater(sum(i=='n' and owner=='points' for i,owner,_,_ in pool.calls),2)
        self.assertTrue(c.last_verification['reached'])
        self.assertEqual(c.status['active_vla_id'],'n')
        self.assertEqual(c.tick_period,1/80)
        self.assertIsNone(c.status['error'])
        c.stop();wait_for(lambda:not c.busy())

    def test_same_time_switch_is_applied_before_point_takeover(self):
        c,_=self.controller();p=multi_plan()
        p['continuous_vla']['takeovers']=plan()['continuous_vla']['takeovers']
        c.start(p,home_before=False)
        wait_for(lambda:any('自动交还' in e['message'] for e in c.events))
        events=[e['message'] for e in c.events]
        switch=next(i for i,e in enumerate(events) if e.startswith('定时 VLA 切换'))
        point=next(i for i,e in enumerate(events) if e.startswith('定时关键点接管'))
        self.assertLess(switch,point)
        self.assertEqual(c.status['active_vla_id'],'n')
        c.stop();wait_for(lambda:not c.busy())

    def test_pause_freezes_model_switch_time(self):
        c,pool=self.controller();p=multi_plan();p['continuous_vla']['switches'][0]['at_s']=.4
        c.start(p,home_before=False);wait_for(lambda:c.status['actions']>1)
        c.pause();time.sleep(.08);elapsed=c.continuous_elapsed
        time.sleep(.3)
        self.assertEqual(c.status['active_vla_id'],'m')
        self.assertEqual(c.continuous_elapsed,elapsed)
        c.resume();wait_for(lambda:c.status['active_vla_id']=='n')
        c.stop();wait_for(lambda:not c.busy())

    def test_unready_next_model_holds_until_fresh_actions_arrive(self):
        c,pool=self.controller();p=multi_plan();p['continuous_vla']['switches'][0]['at_s']=.25
        c.start(p,home_before=False);wait_for(lambda:'n' in pool.models)
        pool.models['n'].stage='warming'
        wait_for(lambda:c.status['active_vla_id']=='n');time.sleep(.1)
        count=len(c.hardware.commands);time.sleep(.1)
        self.assertEqual(len(c.hardware.commands),count)
        self.assertIsNone(c.continuous_policy.desired)
        pool.models['n'].stage='ready'
        wait_for(lambda:c.continuous_policy.desired is not None)
        c.stop();wait_for(lambda:not c.busy())

    def test_failed_next_model_stops_and_holds(self):
        c,pool=self.controller();p=multi_plan();p['continuous_vla']['switches'][0]['at_s']=.2
        c.start(p,home_before=False);wait_for(lambda:'n' in pool.models)
        pool.models['n'].stage='failed';pool.models['n'].detail='second weights unavailable'
        wait_for(lambda:not c.busy())
        self.assertEqual(c.status['state'],'failed')
        self.assertIn('second weights unavailable',c.status['error'])

    def test_late_failure_from_previous_model_is_not_used_after_switch(self):
        c,pool=self.controller();p=multi_plan();p['models'][0]['rtc_queue_threshold']=40
        pool.fail_late=True
        c.start(p,home_before=False)
        wait_for(lambda:c.status['active_vla_id']=='n');time.sleep(.3)
        self.assertTrue(c.busy());self.assertIsNone(c.status['error'])
        self.assertTrue(any(i=='n' for i,_,_,_ in pool.calls))
        c.stop();wait_for(lambda:not c.busy())

    def test_switch_between_joint_and_pose_resets_motion_controller(self):
        c,_=self.controller();p=multi_plan();p['models'][1]['action_mode']='absolute_base_pose'
        p['continuous_vla']['switches'].append(dict(id='return',at_s=.4,model='m'))
        c.start(p,home_before=False)
        wait_for(lambda:any(space=='pose' for space,_ in c.hardware.commands))
        wait_for(lambda:c.status['active_vla_id']=='m' and c.space=='joint')
        self.assertTrue(c.busy());self.assertIsNone(c.status['error'])
        c.stop();wait_for(lambda:not c.busy())

    def test_switch_validation_sorting_save_load_and_deleted_model_cleanup(self):
        p=multi_plan();p['continuous_vla']['switches'].insert(0,dict(id='last',at_s=1,model='m'))
        self.assertEqual([s['at_s'] for s in validate(p)['continuous_vla']['switches']],[.1,1])
        self.assertEqual(used_model_ids(p),{'m','n'})
        for value in [-1,float('nan'),float('inf')]:
            bad=copy.deepcopy(p);bad['continuous_vla']['switches'][0]['at_s']=value
            with self.assertRaises(ValueError):validate(bad)
        for changes in [dict(model='missing'),dict(task=[]),dict(at_s=.1),dict(id='switch1')]:
            bad=copy.deepcopy(p);bad['continuous_vla']['switches'][0].update(changes)
            with self.assertRaises(ValueError):validate(bad)
        with tempfile.TemporaryDirectory() as directory:
            store=SharedPlanStore(directory);identifier=store.save(p)
            self.assertEqual(store.load(identifier)['continuous_vla'],validate(p)['continuous_vla'])
            store.delete_resource('models','n')
            remaining=store.load(identifier)
            self.assertEqual([s['model'] for s in remaining['continuous_vla']['switches']],['m'])
            self.assertEqual(remaining['mode'],'continuous_vla')


if __name__=='__main__':unittest.main()
