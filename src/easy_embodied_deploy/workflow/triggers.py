"""One-shot point takeovers evaluated only while VLA owns the controller."""

import numpy as np


def right_flange_xyz(joints):
    """Piper modified-DH flange position (metres), from measured radians.

    Same nominal geometry as Piper SDK / recorded flange feedback. No TCP
    offset, camera frame or orientation threshold is applied.
    """
    mdh = ((.123, 0, 0, 0), (0, 0, -np.pi/2, -3.0058060377846343),
           (0, .28503, 0, -1.793849405199772), (.25075, -.02198, np.pi/2, 0),
           (0, 0, -np.pi/2, 0), (.091, 0, np.pi/2, 0))
    transform = np.eye(4)
    for angle, (d, a, alpha, offset) in zip(joints[:6], mdh):
        c, s = np.cos(angle+offset), np.sin(angle+offset)
        ca, sa = np.cos(alpha), np.sin(alpha)
        transform = transform @ np.array([[c, -s, 0, a], [ca*s, ca*c, -sa, -sa*d],
                                          [sa*s, sa*c, ca, ca*d], [0, 0, 0, 1]])
    return transform[:3, 3]


class TakeoverSchedule:
    def __init__(self, events, observe=None):
        self.observe = observe
        self.events = events
        self.states = {e['id']: dict(id=e['id'], name=e.get('name', ''), state='waiting') for e in events}
        self.baselines = {}
        self.since = {}
        self.last_sample = None
        self.messages = []

    def interrupt(self):
        # Pauses and other point groups cannot count towards the dwell time.
        self.since.clear()
        self.last_sample = None
        for state in self.states.values():
            if state['state'] == 'matching':
                state.update(state='watching', held_s=0)

    def poll(self, elapsed, read_right_joints):
        active = []
        ready = []
        measured = None
        changed = False
        for event in self.events:
            state = self.states[event['id']]
            if state['state'] in ('triggered', 'expired'):
                continue
            trigger = event.get('trigger', {'type': 'time'})
            if trigger['type'] != 'time' and elapsed > trigger['end_s']:
                changed = True
                state.update(state='expired', held_s=0)
                self.since.pop(event['id'], None)
                self.messages.append(f"关节条件接管已过期：{event.get('name') or event['id']}；未执行固定动作，VLA 继续运行")
            elif elapsed >= event['at_s']:
                if trigger['type'] == 'time':
                    ready.append(event)
                else:
                    active.append(event)

        # No feedback reads outside windows. Read one fresh right-arm sample
        # for every active condition; hardware validates all six CAN groups.
        if active and (self.last_sample is None or elapsed-self.last_sample >= .04):
            joints = np.asarray(read_right_joints(), dtype=float)
            if joints.shape != (7,) or not np.all(np.isfinite(joints)):
                raise RuntimeError('关节触发需要完整、有效的右臂六轴与夹爪反馈')
            if self.last_sample is not None and elapsed-self.last_sample > .2:
                self.since.clear()
            self.last_sample = elapsed
            measured = np.rad2deg(joints[:6]).tolist()
            angles = np.rad2deg(joints[[3, 5]])
            for event in active:
                key, trigger = event['id'], event['trigger']
                if trigger['type'] in ('right_j6_pose', 'right_joint_sum_pose'):
                    xyz = right_flange_xyz(joints)
                    j4, j6 = float(angles[0]), float(angles[1])
                    z_mm = float(xyz[2]*1000)
                    z_matched = bool(z_mm+1e-9 >= trigger['z_mm'])
                    self.states[key].update(type=trigger['type'], j4_deg=j4, j6_deg=j6, z_mm=z_mm,
                        xyz_mm=(xyz*1000).tolist(), z_threshold_mm=trigger['z_mm'], z_matched=z_matched)
                    if trigger['type'] == 'right_joint_sum_pose':
                        total = j4+j6
                        angle_matched = bool(total+1e-9 >= trigger['sum_deg'])
                        self.states[key].update(sum_deg=total, sum_threshold_deg=trigger['sum_deg'],
                                                sum_matched=angle_matched)
                    else:
                        angle_matched = bool(j6+1e-9 >= trigger['j6_deg'])
                        self.states[key].update(j6_threshold_deg=trigger['j6_deg'], j6_matched=angle_matched)
                    combined_matched = angle_matched and z_matched
                    high_z = trigger.get('high_z_mm') if trigger['type'] == 'right_joint_sum_pose' else None
                    high_matched = bool(high_z is not None and z_mm+1e-9 >= high_z)
                    matched = combined_matched or high_matched
                    if trigger['type'] == 'right_joint_sum_pose':
                        self.states[key].update(combined_matched=combined_matched,
                            high_z_threshold_mm=high_z, high_z_matched=high_matched,
                            matched_by=('both' if combined_matched and high_matched else
                                        'sum_and_height' if combined_matched else 'high_z' if high_matched else None))
                else:
                    if key not in self.baselines:
                        self.baselines[key] = (angles.copy() if trigger['reference'] == 'window' else
                                               np.array([trigger['j4_reference_deg'], trigger['j6_reference_deg']]))
                    baseline = self.baselines[key]
                    # Bounded physical joint coordinates; no angular wrapping.
                    delta = angles-baseline
                    total = float(np.sum(delta))
                    matched = bool(total+1e-9 >= trigger['sum_deg'])
                    self.states[key].update(j4_deg=float(angles[0]), j6_deg=float(angles[1]), sum_deg=total,
                        baseline_deg=baseline.tolist(), delta_deg=delta.tolist())
                if matched:
                    self.since.setdefault(key, elapsed)
                else:
                    self.since.pop(key, None)
                held = max(0., elapsed-self.since[key]) if key in self.since else 0.
                self.states[key].update(state='matching' if matched else 'watching',
                    held_s=held)
                if matched and held+1e-9 >= trigger['hold_s']:
                    ready.append(event)

        if not ready:
            if self.observe and (measured is not None or changed):
                self.observe(dict(schedule_elapsed=float(elapsed), right_joints_deg=measured,
                                  conditions=self.snapshot()))
            return None
        # A waiting conditional event must not block later timed/conditional
        # events. Select one owner; never queue a stale conditional match.
        event = min(ready, key=lambda e: self.events.index(e))
        self.states[event['id']].update(state='triggered', triggered_at_s=float(elapsed))
        if self.observe:
            self.observe(dict(schedule_elapsed=float(elapsed), right_joints_deg=measured, conditions=self.snapshot()))
        self.interrupt()
        return event

    def snapshot(self):
        return [dict(self.states[e['id']]) for e in self.events]
