"""Read-only 20 Hz feedback capture; disk I/O never runs on the control loop."""

import copy
import csv
import io
import json
import math
import queue
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .triggers import right_flange_xyz

RUN_ID = re.compile(r'^\d{8}T\d{6}Z-[a-f0-9]{8}$')


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2))
    temporary.replace(path)


class MotionRecording:
    def __init__(self, root, plan, feedback, context, simulated, period=.05):
        self.identifier = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-')+uuid.uuid4().hex[:8]
        self.path = Path(root)/self.identifier
        self.path.mkdir(parents=True)
        self.feedback, self.context, self.period = feedback, context, period
        self.started_wall, self.started_mono = time.time(), time.monotonic()
        self.pending = queue.Queue(maxsize=4096)
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.status = dict(id=self.identifier, path=str(self.path), running=True, samples=0,
                           feedback_errors=0, dropped_events=0, error=None)
        self.metadata = dict(schema='workflow_motion_recording', version=1, id=self.identifier,
            started_at=self.started_wall, name=plan.get('name', ''), plan=copy.deepcopy(plan),
            simulated=bool(simulated), nominal_hz=1/period,
            joint_order=['left_J'+str(i) for i in range(1,7)]+['right_J'+str(i) for i in range(1,7)],
            units=dict(joints_deg='degrees', flange_xyz_mm='mm, each arm base frame, nominal Piper FK; no TCP offset'),
            clocks='wall_time: Unix seconds; elapsed_s: wall duration since run start; schedule_elapsed: pausable VLA clock')
        write_json(self.path/'metadata.json', self.metadata)
        self.thread = threading.Thread(target=self._run, name='workflow-recording', daemon=True)
        self.thread.start()

    def snapshot(self):
        with self.lock:
            return dict(self.status)

    def emit(self, kind, payload):
        # Only bounded queue insertion on motion/inference threads, never disk.
        row = dict(kind=kind, wall_time=time.time(), elapsed_s=time.monotonic()-self.started_mono,
                   data=copy.deepcopy(payload))
        try:
            self.pending.put_nowait(row)
        except queue.Full:
            with self.lock:
                self.status['dropped_events'] += 1

    def finish(self, result):
        self.emit('run_end', result)
        self.stop_event.set()
        self.thread.join(timeout=2)
        if self.thread.is_alive():
            with self.lock:
                self.status['error'] = '记录线程尚未结束，请检查磁盘；原始文件保留'

    def _run(self):
        try:
            with (self.path/'samples.jsonl').open('w', buffering=1) as samples, \
                 (self.path/'events.jsonl').open('w', buffering=1) as events:
                next_sample = time.monotonic()
                while not self.stop_event.is_set() or not self.pending.empty():
                    now = time.monotonic()
                    if not self.stop_event.is_set() and now >= next_sample:
                        row = dict(wall_time=time.time(), elapsed_s=now-self.started_mono,
                                   **self.context())
                        try:
                            feedback = self.feedback()
                            q = np.asarray(feedback['joints_rad'], dtype=float)
                            if q.shape != (12,) or not np.all(np.isfinite(q)):
                                raise ValueError('完整十二轴反馈无效')
                            row.update(joints_deg=np.rad2deg(q).tolist(),
                                flange_xyz_mm={side:(right_flange_xyz(q[offset:offset+6])*1000).tolist()
                                               for side,offset in [('left',0),('right',6)]},
                                feedback=feedback.get('provenance', {}))
                        except Exception as exc:
                            # A capture gap must never become a repeated cached pose.
                            row['feedback_error'] = str(exc)
                            with self.lock:
                                self.status['feedback_errors'] += 1
                        samples.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                        with self.lock:
                            self.status['samples'] += 1
                        next_sample = time.monotonic()+self.period
                    for _ in range(256):
                        try:
                            item = self.pending.get_nowait()
                        except queue.Empty:
                            break
                        events.write(json.dumps(item, ensure_ascii=False, allow_nan=False)+'\n')
                    if not self.stop_event.is_set():
                        self.stop_event.wait(min(.02, max(0, next_sample-time.monotonic())))
            with self.lock:
                self.status['running'] = False
                summary = dict(self.status, ended_at=time.time(), duration_s=time.monotonic()-self.started_mono)
            write_json(self.path/'summary.json', summary)
        except Exception as exc:
            with self.lock:
                self.status.update(running=False, error=f'运行记录保存失败：{exc}')
            try:
                write_json(self.path/'summary.json', self.snapshot())
            except OSError:
                pass


def run_path(root, identifier):
    if not isinstance(identifier, str) or not RUN_ID.fullmatch(identifier):
        raise ValueError('无效运行记录标识')
    root = Path(root).resolve()
    path = (root/identifier).resolve()
    if path.parent != root or not path.is_dir():
        raise ValueError('运行记录不存在')
    return path


def read_lines(path):
    if not path.exists():
        return []
    rows=[]
    with path.open() as stream:
        for line in stream:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # A live writer can leave its final line temporarily incomplete.
                continue
    return rows


def recordings(root):
    root=Path(root)
    if not root.exists():
        return []
    rows=[]
    for folder in sorted(root.iterdir(),reverse=True):
        if not RUN_ID.fullmatch(folder.name):
            continue
        try:
            folder=run_path(root,folder.name)
            m=json.loads((folder/'metadata.json').read_text())
            summary=json.loads((folder/'summary.json').read_text()) if (folder/'summary.json').exists() else {}
            rows.append(dict(id=folder.name,name=m['name'],started_at=m['started_at'],simulated=m['simulated'],
                             samples=summary.get('samples'),finished=bool(summary),error=summary.get('error')))
        except (OSError,ValueError,KeyError):
            continue
        if len(rows)>=100:
            break
    return sorted(rows, key=lambda r:r['started_at'], reverse=True)


def recording_data(root, identifier):
    folder=run_path(root,identifier)
    samples=read_lines(folder/'samples.jsonl')
    stride=max(1,math.ceil(len(samples)/20000))
    view=samples[::stride]
    if samples and (len(samples)-1)%stride:
        view.append(samples[-1])
    return dict(metadata=json.loads((folder/'metadata.json').read_text()), samples=view,
                summary=json.loads((folder/'summary.json').read_text()) if (folder/'summary.json').exists() else None,
                events=read_lines(folder/'events.jsonl'),total_samples=len(samples),display_stride=stride)


def samples_csv(root, identifier):
    folder=run_path(root,identifier)
    output=io.StringIO();writer=csv.writer(output)
    writer.writerow(['wall_time','elapsed_s','schedule_elapsed','state','controller_owner']+
                    [f'{side}_J{i}_deg' for side in ['left','right'] for i in range(1,7)]+
                    [f'{side}_{axis}_mm' for side in ['left','right'] for axis in 'XYZ']+['feedback_error'])
    for row in read_lines(folder/'samples.jsonl'):
        xyz=row.get('flange_xyz_mm',{})
        writer.writerow([row.get(k,'') for k in ['wall_time','elapsed_s','schedule_elapsed','state','controller_owner']]+
                        row.get('joints_deg',['']*12)+xyz.get('left',['']*3)+xyz.get('right',['']*3)+[row.get('feedback_error','')])
    return ('\ufeff'+output.getvalue()).encode()
