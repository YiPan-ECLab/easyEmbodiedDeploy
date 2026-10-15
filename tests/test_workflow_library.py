"""Shared resources survive new flows, stale drafts, restarts and concurrent writers."""

import copy
import json
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from easy_embodied_deploy import recording_start
from easy_embodied_deploy.workflow.library import SharedPlanStore
from easy_embodied_deploy.workflow.plan import PlanStore, new_plan


def example():
    plan = new_plan()
    plan['points'] = [dict(id='p', name='基准点', space='joint', arms={'left': [0]*7})]
    plan['models'] = [dict(id='m', name='模型', backend='mock', action_mode='absolute_joint')]
    plan['steps'] = [dict(id='s1', type='point', point='p'), dict(id='s2', type='vla', model='m')]
    return plan


class SharedLibraryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = SharedPlanStore(self.temp.name)

    def test_new_flow_and_restart_keep_resources_without_saving_workflow(self):
        plan = example()
        for kind in ('points', 'models'):
            self.store.save_resource(kind, plan[kind][0])
        other = SharedPlanStore(self.temp.name)
        fresh = other.resolve(new_plan())
        self.assertEqual(fresh['steps'], [])
        self.assertEqual(fresh['points'][0]['name'], '基准点')
        self.assertEqual(fresh['models'][0]['name'], '模型')

    def test_edits_shared_by_all_flows_and_stale_save_cannot_revert(self):
        first = self.store.save(example())
        stale = self.store.load(first)
        second = self.store.save(stale)
        point = copy.deepcopy(stale['points'][0])
        point['arms']['left'][0] = .123
        model = dict(stale['models'][0], name='新模型名', policy_hz=25)
        self.store.save_resource('points', point)
        self.store.save_resource('models', model)
        self.store.save(stale, first)
        for identifier in (first, second):
            current = self.store.load(identifier)
            self.assertEqual(current['points'][0]['arms']['left'][0], .123)
            self.assertEqual(current['models'][0]['policy_hz'], 25)
            disk = json.loads(self.store.path(identifier).read_text())
            self.assertEqual(disk['points'], [])
            self.assertEqual(disk['models'], [])
            self.assertEqual(disk['steps'][0]['point'], 'p')

    def test_delete_updates_all_references_and_old_draft_does_not_resurrect(self):
        first = self.store.save(example())
        old = self.store.load(first)
        second = self.store.save(old)
        self.store.delete_resource('points', 'p')
        self.store.delete_resource('models', 'm')
        self.store.save(old, first)
        for identifier in (first, second):
            current = self.store.load(identifier)
            self.assertEqual(current['steps'], [])
            self.assertEqual(current['points'], [])
            self.assertEqual(current['models'], [])
        self.assertEqual(SharedPlanStore(self.temp.name).snapshot()['models'], [])

    def test_legacy_migration_preserves_backup_and_prefers_existing_library(self):
        plan = example()
        identifier = PlanStore(self.temp.name).save(plan)
        original = self.store.path(identifier).read_text()
        point = dict(plan['points'][0], name='已更新共享点')
        self.store.save_resource('points', point)
        migrated = SharedPlanStore(self.temp.name)
        self.assertEqual(migrated.load(identifier)['points'][0]['name'], '已更新共享点')
        self.assertEqual(migrated.snapshot()['models'][0]['id'], 'm')
        backup = Path(self.temp.name) / 'legacy_snapshots' / f'{identifier}.json'
        self.assertEqual(json.loads(backup.read_text()), json.loads(original))

    def test_independent_servers_do_not_lose_concurrent_resources(self):
        second = SharedPlanStore(self.temp.name)
        def save(index):
            store = self.store if index % 2 else second
            store.save_resource('points', dict(example()['points'][0], id=f'p{index}'))
        with ThreadPoolExecutor(max_workers=4) as workers:
            list(workers.map(save, range(20)))
        self.assertEqual(len(self.store.snapshot()['points']), 20)

    def test_recorder_import_after_migration_and_saved_start_is_independent(self):
        root = Path(self.temp.name)
        directory = root / 'datasets/workflows'
        plan = example()
        plan['points'][0]['arms'] = dict(left=[.1, .2, -.3, 0, 0, 0, .04],
                                        right=[-.1, .3, -.4, 0, 0, 0, .03])
        old = PlanStore(directory)
        old.save(plan)
        old.save(plan)
        with patch.object(recording_start, 'ROOT', root), patch.object(recording_start, 'START_PATH', root/'start.json'):
            recording_start.save(dict(mode='custom', name='原采集起点', arms=plan['points'][0]['arms']))
            before = recording_start.START_PATH.read_bytes()
            shared = SharedPlanStore(directory)
            candidates = recording_start.workflow_points()
            self.assertEqual(len(candidates), 1)  # no VLA rows or per-workflow duplicates
            self.assertEqual(candidates[0]['arms'], plan['points'][0]['arms'])
            self.assertEqual(recording_start.normalize(dict(mode='custom', **candidates[0]))['arms'], plan['points'][0]['arms'])
            updated = copy.deepcopy(plan['points'][0])
            updated['arms']['right'][0] = .25
            shared.save_resource('points', updated)
            self.assertEqual(recording_start.workflow_points()[0]['arms']['right'][0], .25)
            self.assertEqual(recording_start.START_PATH.read_bytes(), before)
            self.assertEqual(recording_start.load()['arms'], plan['points'][0]['arms'])


if __name__ == '__main__':
    unittest.main()
