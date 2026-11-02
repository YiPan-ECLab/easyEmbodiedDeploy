"""RTC alignment under capture overhead, poll jitter and genuine overruns."""

import json
import sys
import tempfile
import threading
import time
import unittest
from collections import deque
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from easy_embodied_deploy.workflow.rtc_timing import RTCTiming
from easy_embodied_deploy.workflow.continuous import ContinuousPolicy
from easy_embodied_deploy.workflow.runtime import WorkflowController
from easy_embodied_deploy.workflow.models import ModelProcess
from easy_embodied_deploy.workflow.plan import new_plan


def model():
    return SimpleNamespace(config=dict(id='m',name='RTC model',policy_hz=30,control_hz=100,
        inference_mode='rtc',rtc_mode='trained',rtc_horizon=10,rtc_queue_threshold=30,
        action_mode='absolute_joint',smoothing_time_constant=0),
        inference={'training_max_delay':10,'action_steps':50},latencies=[.14],stage='ready')


def result_future(requested=0,finished=.14):
    future=Future()
    future._workflow_timing=dict(requested_at=requested,dispatched_at=requested+.04,finished_at=finished)
    future.set_result(np.arange(50*14,dtype=float).reshape(50,14))
    return future


class TimingTests(unittest.TestCase):
    def test_poll_delay_does_not_make_a_covered_prefix_expire(self):
        tracker=RTCTiming(model(),30,.01)
        with patch('time.monotonic',return_value=.2):
            report=tracker.complete(result_future(),5,5,True,0)
        self.assertEqual(report['wall_frames'],6)
        self.assertEqual(report['inference_frames'],5)
        self.assertEqual(report['consumed_frames'],5)
        self.assertFalse(report['discarded'])
        self.assertAlmostEqual(report['poll_delay_ms'],60)

    def test_capture_and_dispatch_overhead_adapts_budget_but_cannot_exceed_training(self):
        m=model();m.latencies=[.05]
        tracker=RTCTiming(m,30,.01);before=tracker.reserve(30)
        with patch('time.monotonic',return_value=.24):
            tracker.complete(result_future(0,.23),4,6,True,0)
        self.assertGreater(tracker.reserve(30),before)
        self.assertEqual(tracker.reserve(3),3)
        m.latencies=[2]
        self.assertEqual(tracker.reserve(50),10)
        self.assertEqual(tracker.reserve(0),0)

    def test_genuinely_executed_unconditioned_actions_are_rejected(self):
        with patch('time.monotonic',return_value=.2):
            report=RTCTiming(model(),30,.01).complete(result_future(),6,5,True,0)
        self.assertTrue(report['discarded']);self.assertEqual(report['reason'],'prefix_exhausted')

    def test_real_request_over_training_limit_is_rejected_even_if_queue_starved(self):
        with patch('time.monotonic',return_value=.5):
            report=RTCTiming(model(),30,.01).complete(result_future(0,.4),3,5,True,0)
        self.assertTrue(report['discarded']);self.assertEqual(report['reason'],'training_limit_exceeded')

    def test_unconditioned_startup_and_guided_rtc_keep_existing_semantics(self):
        tracker=RTCTiming(model(),30,.01)
        with patch('time.monotonic',return_value=1):
            self.assertFalse(tracker.complete(result_future(0,1),0,0,False,0)['discarded'])
        m=model();m.config['rtc_mode']='guided';tracker=RTCTiming(m,30,.01)
        self.assertEqual(tracker.reserve(30),5)
        with patch('time.monotonic',return_value=1):
            self.assertFalse(tracker.complete(result_future(0,1),20,5,True,0)['discarded'])

    def test_continuous_mode_survives_five_wall_clock_false_positives(self):
        m=model();reports=[]
        controller=SimpleNamespace(models=SimpleNamespace(get=lambda _:m),
            _submit_model=lambda *a:Future(),_record_rtc=lambda r,c:reports.append(r))
        policy=ContinuousPolicy(controller,{'model':'m'})
        policy.next_policy=float('inf')
        for _ in range(6):
            policy.pending=result_future();policy.pending_prefix=True
            policy.pending_consumed=5;policy.conditioned_delay=5;policy.submitted_at=0
            with patch('time.monotonic',return_value=.2):policy.tick()
        self.assertEqual(policy.discarded,0)
        self.assertEqual(len(reports),6)
        np.testing.assert_array_equal(policy.queue[0],np.arange(50*14).reshape(50,14)[5])

    def test_continuous_mode_still_stops_after_five_true_prefix_overruns(self):
        m=model();controller=SimpleNamespace(models=SimpleNamespace(get=lambda _:m),_submit_model=lambda *a:Future())
        policy=ContinuousPolicy(controller,{'model':'m'});policy.next_policy=float('inf')
        for index in range(5):
            policy.pending=result_future();policy.pending_prefix=True
            policy.pending_consumed=6;policy.conditioned_delay=5;policy.submitted_at=0
            with patch('time.monotonic',return_value=.2):
                if index==4:
                    with self.assertRaisesRegex(RuntimeError,'已执行 6 帧 / 预留 5 帧'):policy.tick()
                else:policy.tick()

    def test_sequential_mode_uses_action_consumption_despite_control_loop_jitter(self):
        m=model();m.config['rtc_queue_threshold']=50
        pool=Mock();pool.get.return_value=m
        c=WorkflowController(pool);self.addCleanup(c.close);c.connect(True)
        c.plan=new_plan();c.plan['steps']=[dict(id='v',type='vla',model='m',until='duration',value=2)]
        clock=SimpleNamespace(now=100.)
        futures=[]
        class ScheduledFuture:
            def __init__(self,prefix):
                self._workflow_timing=dict(requested_at=clock.now,finished_at=clock.now+(.14 if prefix is not None else 0))
                self.delayed=False
            def done(self):return clock.now>=self._workflow_timing['finished_at']
            def result(self):return np.tile(c.hardware.joints,(50,1))
        def submit(index,prefix=None,delay=0):
            future=ScheduledFuture(prefix);futures.append(future);return future
        def sleep(start):
            before=clock.now;clock.now+=c.tick_period
            for future in futures:
                end=future._workflow_timing['finished_at']
                if before<end<=clock.now and not future.delayed:
                    clock.now+=.1;future.delayed=True
        with patch('time.monotonic',side_effect=lambda:clock.now), \
             patch.object(c,'_submit',side_effect=submit),patch.object(c,'_sleep',side_effect=sleep):
            c._vla(0)
        self.assertGreater(c.status['actions'],20)
        self.assertGreater(len(c.rtc_diagnostics),5)
        self.assertFalse(any(r['discarded'] for r in c.rtc_diagnostics))
        self.assertTrue(any(r['wall_frames']>r['reserved_frames'] and r['consumed_frames']<=r['reserved_frames']
                            for r in c.rtc_diagnostics if r['reserved_frames']))

    def test_future_times_cover_observation_and_complete_before_poll(self):
        c=WorkflowController();self.addCleanup(c.close);c.connect(True)
        def observe(model):time.sleep(.005);return {}
        def infer(*a,**kw):time.sleep(.005);return np.zeros((50,14))
        with patch.object(c,'_observation',side_effect=observe):
            future=c._submit_model(SimpleNamespace(infer=infer),'')
            future.result(timeout=1)
        timing=future._workflow_timing
        self.assertGreater(timing['dispatched_at']-timing['requested_at'],.004)
        self.assertGreater(timing['finished_at']-timing['requested_at'],.008)
        self.assertLessEqual(timing['finished_at'],time.monotonic())

    def test_diagnostics_persist_without_observation_or_action_payloads(self):
        c=WorkflowController();self.addCleanup(c.close);c.plan=new_plan()
        c._record_rtc({'discarded':False,'request_ms':150,'consumed_frames':4,'reserved_frames':6},0)
        with tempfile.TemporaryDirectory() as directory,patch('easy_embodied_deploy.module_http.ROOT',Path(directory)):
            c._save_rtc_diagnostics()
            data=json.loads(Path(c.rtc_log_path).read_text())
            self.assertEqual(data['samples'][0]['request_ms'],150)
            self.assertNotIn('observation',data['samples'][0])

    def test_model_snapshot_exposes_latency_and_training_cap_without_credentials(self):
        m=ModelProcess.__new__(ModelProcess)
        m.config={'id':'m'};m.lock=threading.RLock();m.stage='ready';m.detail='';m.started=time.monotonic()
        m.process=SimpleNamespace(pid=123);m.logs=[];m.inference={'training_max_delay':10};m.latencies=deque([.1,.2])
        m.auth=b'private'
        snapshot=m.snapshot()
        self.assertEqual(snapshot['rpc_latency_ms']['peak'],200)
        self.assertEqual(snapshot['inference']['training_max_delay'],10)
        self.assertNotIn('auth',snapshot)


if __name__=='__main__':unittest.main()
