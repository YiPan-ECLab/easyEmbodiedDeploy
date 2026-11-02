"""Capture actual feedback and exact decisions without a second motion owner."""
import json
import queue
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from test_workflow_continuous import plan, Pool, wait_for
from test_workflow import point_plan
from easy_embodied_deploy.workflow.hardware import MockHardware, PiperHardware
from easy_embodied_deploy.workflow.runtime import WorkflowController
from easy_embodied_deploy.workflow.recording import (MotionRecording, recordings, recording_data,
                                                    samples_csv, run_path)
from easy_embodied_deploy.workflow.triggers import TakeoverSchedule
from easy_embodied_deploy.workflow.plan import validate


class RecordingTests(unittest.TestCase):
    def make(self, directory, feedback=None, context=None):
        hw=MockHardware()
        return MotionRecording(directory,plan(),feedback or hw.motion_feedback,
            context or (lambda:dict(state='running',schedule_elapsed=1,controller_owner='vla')),True,period=.01)

    def test_feedback_units_configuration_and_independent_history(self):
        with tempfile.TemporaryDirectory() as d:
            q=np.array([0,.5,-1.5,0,0,np.pi/6]*2)
            r=self.make(d,lambda:dict(joints_rad=q.copy()))
            wait_for(lambda:r.snapshot()['samples']>=2)
            r.finish(dict(state='stopped'))
            data=recording_data(d,r.identifier)
            self.assertAlmostEqual(data['samples'][0]['joints_deg'][11],30)
            self.assertGreater(data['samples'][0]['flange_xyz_mm']['right'][2],500)
            self.assertEqual(data['metadata']['plan']['continuous_vla']['model'],'m')
            self.assertFalse(data['summary']['running'])
            self.assertEqual(data['events'][-1]['kind'],'run_end')
            second=self.make(d);wait_for(lambda:second.snapshot()['samples']>0);second.finish(dict(state='completed'))
            self.assertEqual(len(recordings(d)),2)
            self.assertTrue((r.path/'samples.jsonl').exists())
            self.assertIn('right_J6_deg',samples_csv(d,r.identifier).decode('utf-8-sig'))

    def test_missing_or_invalid_feedback_records_gap_not_cached_angles(self):
        with tempfile.TemporaryDirectory() as d:
            n=[0]
            def feedback():
                n[0]+=1
                if n[0]==1:return dict(joints_rad=[0]*12)
                raise RuntimeError('stale CAN')
            r=self.make(d,feedback);wait_for(lambda:r.snapshot()['feedback_errors']>=2);r.finish(dict(state='stopped'))
            data=recording_data(d,r.identifier)
            self.assertIn('joints_deg',data['samples'][0])
            for row in data['samples'][1:]:
                self.assertEqual(row['feedback_error'],'stale CAN');self.assertNotIn('joints_deg',row)

    def test_live_reader_partial_line_and_path_traversal(self):
        with tempfile.TemporaryDirectory() as d:
            r=self.make(d);wait_for(lambda:r.snapshot()['samples']>0);r.finish(dict(state='stopped'))
            with (r.path/'samples.jsonl').open('a') as f:f.write('{"incomplete":')
            self.assertGreater(recording_data(d,r.identifier)['total_samples'],0)
            for invalid in ['../metadata.json','/etc/passwd','x',r.identifier+'/..']:
                with self.assertRaises(ValueError):run_path(d,invalid)
            with tempfile.TemporaryDirectory() as outside:
                fake=r.identifier[:-8]+'deadbeef';(Path(d)/fake).symlink_to(outside)
                with self.assertRaises(ValueError):run_path(d,fake)

    def test_events_are_frozen_and_overflow_is_visible(self):
        with tempfile.TemporaryDirectory() as d:
            r=self.make(d);payload=dict(value=[1]);r.emit('condition',payload);payload['value'][0]=99
            r.finish(dict(state='stopped'))
            data=recording_data(d,r.identifier);self.assertEqual(data['events'][0]['data']['value'],[1])
            r.pending=queue.Queue(maxsize=1);r.emit('a',{});r.emit('b',{})
            self.assertEqual(r.snapshot()['dropped_events'],1)

    def test_storage_failure_does_not_escape_or_change_hardware(self):
        with tempfile.TemporaryDirectory() as d:
            # A failed writer must be visible and cannot issue any command.
            hw=MockHardware()
            original=Path.open
            def fail_samples(path,*args,**kwargs):
                if path.name=='samples.jsonl':raise OSError('disk full')
                return original(path,*args,**kwargs)
            with patch.object(Path,'open',fail_samples):
                r=self.make(d,hw.motion_feedback);r.thread.join(1)
            self.assertIn('disk full',r.snapshot()['error'])
            self.assertEqual(hw.commands,[])

    def test_read_only_feedback_fails_fast_on_missing_can(self):
        hw=object.__new__(PiperHardware)
        hw.arms={'right':SimpleNamespace(_parser=SimpleNamespace())}
        before=time.monotonic()
        with self.assertRaises(RuntimeError):hw._joint_sample('right',wait=False)
        self.assertLess(time.monotonic()-before,.05)

    def test_condition_observer_keeps_actual_feedback_thresholds_and_expiry(self):
        p=plan();e=p['continuous_vla']['takeovers'][0];e.update(at_s=1,trigger=dict(type='right_j6_pose',end_s=2,j6_deg=30,z_mm=250,hold_s=.2))
        seen=[];s=TakeoverSchedule(validate(p)['continuous_vla']['takeovers'],observe=seen.append)
        s.poll(1,lambda:[0,.2,-.2,0,0,np.deg2rad(-17),.04])
        c=seen[-1]['conditions'][0]
        self.assertFalse(c['j6_matched']);self.assertTrue(c['z_matched'])
        self.assertAlmostEqual(seen[-1]['right_joints_deg'][5],-17)
        self.assertEqual(c['j6_threshold_deg'],30)
        s.poll(2.01,lambda:(_ for _ in ()).throw(AssertionError('out of window')))
        self.assertEqual(seen[-1]['conditions'][0]['state'],'expired')
        self.assertIsNone(seen[-1]['right_joints_deg'])

    def test_controller_records_pause_takeover_return_and_preserves_previous_run(self):
        with tempfile.TemporaryDirectory() as d,patch('easy_embodied_deploy.module_http.ROOT',Path(d)):
            pool=Pool();c=WorkflowController(pool);pool.controller=c;c.connect(True)
            self.addCleanup(c.close)
            p=plan();p['continuous_vla']['takeovers'][0].update(at_s=.25,
                trigger=dict(type='right_j6_pose',end_s=3,j6_deg=30,z_mm=250,hold_s=.1))
            c.start(p,home_before=False)
            wait_for(lambda:c.status['actions']>0)
            c.pause();time.sleep(.12);c.resume()
            wait_for(lambda:any('自动交还' in e['message'] for e in c.events))
            c.stop();wait_for(lambda:not c.busy())
            first=c.snapshot()['motion_recording'];root=Path(d)/'logs/workflow-runs'
            data=recording_data(root,first['id'])
            self.assertTrue(any(r.get('paused') for r in data['samples']))
            self.assertTrue(any(r.get('controller_owner')=='points' for r in data['samples']))
            self.assertTrue(any(e['kind']=='condition' and e['data']['conditions'][0]['state']=='triggered' for e in data['events']))
            self.assertFalse(first['running']);self.assertIsNone(first['error'])
            c.start(point_plan(),home_before=False);wait_for(lambda:not c.busy())
            self.assertNotEqual(c.snapshot()['motion_recording']['id'],first['id'])
            self.assertTrue(run_path(root,first['id']).exists())

    def test_emergency_and_failed_run_flush_summary(self):
        for emergency in [True,False]:
            with self.subTest(emergency=emergency),tempfile.TemporaryDirectory() as d,patch('easy_embodied_deploy.module_http.ROOT',Path(d)):
                pool=Pool();c=WorkflowController(pool);pool.controller=c;c.connect(True)
                p=plan()
                if emergency:
                    c.start(p,home_before=False);wait_for(lambda:c.status['actions']>0);c.stop(emergency=True)
                else:
                    with patch.object(c.hardware,'prepare',side_effect=RuntimeError('motor fault')):
                        c.start(p,home_before=False);wait_for(lambda:not c.busy())
                wait_for(lambda:not c.busy())
                r=c.snapshot()['motion_recording'];data=recording_data(Path(d)/'logs/workflow-runs',r['id'])
                self.assertIsNotNone(data['summary'])
                self.assertEqual(data['events'][-1]['data']['state'],'emergency' if emergency else 'failed')
                c.close()

    def test_http_listing_replay_exports_and_invalid_paths(self):
        import urllib.request
        import urllib.error
        from easy_embodied_deploy.workflow.console import WorkflowServer
        with tempfile.TemporaryDirectory() as d,patch('easy_embodied_deploy.workflow.console.ROOT',Path(d)):
            root=Path(d)/'logs/workflow-runs'
            r=self.make(root);wait_for(lambda:r.snapshot()['samples']>0);r.finish(dict(state='completed'))
            server=WorkflowServer(('127.0.0.1',0),directory=Path(d)/'library')
            worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
            base='http://127.0.0.1:'+str(server.server_port)
            try:
                listing=json.load(urllib.request.urlopen(base+'/api/recordings'))
                self.assertEqual(listing['runs'][0]['id'],r.identifier)
                data=json.load(urllib.request.urlopen(base+'/api/recording?id='+r.identifier))
                self.assertEqual(len(data['samples'][0]['joints_deg']),12)
                self.assertTrue(data['metadata']['simulated'])
                csv_data=urllib.request.urlopen(base+'/api/recording?id='+r.identifier+'&file=samples.csv').read()
                self.assertTrue(csv_data.startswith(b'\xef\xbb\xbf'))
                self.assertIn(b'right_J6_deg',csv_data)
                raw=urllib.request.urlopen(base+'/api/recording?id='+r.identifier+'&file=events.jsonl').read()
                self.assertIn(b'run_end',raw)
                for suffix in ['/api/recording?id=../x','/api/recording?id='+r.identifier+'&file=../metadata.json']:
                    with self.assertRaises(urllib.error.HTTPError) as error:urllib.request.urlopen(base+suffix)
                    self.assertEqual(error.exception.code,400)
            finally:
                server.shutdown();server.server_close();server.controller.close();worker.join(1)


if __name__=='__main__':unittest.main()
