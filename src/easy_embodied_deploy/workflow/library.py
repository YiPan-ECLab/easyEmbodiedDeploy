"""Shared workflow resources, with process-safe writes and legacy migration."""

import copy
import fcntl
import json
import threading
from contextlib import contextmanager

from ..calibration import atomic_json
from .plan import PlanStore, new_plan, validate, validate_names, unique_name, repair_vla_names


class SharedPlanStore(PlanStore):
    def __init__(self, directory):
        super().__init__(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        with self.transaction():
            self._migrate()

    @contextmanager
    def transaction(self):
        with self.lock, (self.directory / '.library.lock').open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _read(self, filename, default):
        path = self.directory / filename
        return json.loads(path.read_text()) if path.exists() else copy.deepcopy(default)

    def _state(self):
        return self._read('library_state.json', dict(revision=0, deleted_points=[], deleted_models=[]))

    def _touch(self, state=None):
        state = state or self._state()
        state['revision'] += 1
        atomic_json(self.directory / 'library_state.json', state)

    @staticmethod
    def _filename(kind):
        return {'points': 'point_library.json', 'models': 'model_library.json'}[kind]

    def _rows(self, kind):
        deleted = set(self._state()['deleted_' + kind])
        return [row for row in self._read(self._filename(kind), []) if row['id'] not in deleted]

    def _write(self, kind, rows):
        atomic_json(self.directory / self._filename(kind), rows)

    def revision(self):
        return self._state()['revision']

    def snapshot(self):
        with self.transaction():
            return dict(points=self._rows('points'), models=self._rows('models'), **self._state())

    def _documents(self):
        for path in sorted(self.directory.glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                self.path(path.stem)
                value = json.loads(path.read_text())
                if value.get('schema') == new_plan()['schema']:
                    yield path, value
            except (ValueError, TypeError, AttributeError):
                continue

    def _migrate(self):
        # Existing shared points take precedence; newest legacy model snapshots
        # seed the model library. Original workflow documents remain recoverable.
        documents = list(self._documents())
        # Preserve pre-uniqueness libraries before assigning display-name suffixes.
        # IDs, coordinates, model configurations and step references stay intact.
        for kind in ('points', 'models'):
            rows = self._rows(kind)
            upgraded = []
            for row in rows:
                row = copy.deepcopy(row)
                row['name'] = unique_name((row.get('name') or row['id']).strip(), upgraded)
                upgraded.append(row)
            if rows != upgraded:
                backup = self.directory / 'legacy_snapshots' / ('names_' + self._filename(kind))
                backup.parent.mkdir(exist_ok=True)
                if not backup.exists():
                    atomic_json(backup, rows)
                self._write(kind, upgraded)
                self._touch()
        for _, document in documents:
            if document.get('resources') != 'shared':
                self._resolve(document, legacy=True)
        for path, document in documents:
            if document.get('resources') != 'shared':
                backup = self.directory / 'legacy_snapshots' / path.name
                backup.parent.mkdir(exist_ok=True)
                if not backup.exists():
                    atomic_json(backup, document)
                atomic_json(path, self._reference_document(self._resolve(document, legacy=True)))

    @staticmethod
    def _reference_document(plan):
        document = copy.deepcopy(plan)
        document.update(points=[], models=[], resources='shared')
        return document

    def _resolve(self, plan, legacy=False):
        plan = copy.deepcopy(plan)
        state = self._state()
        changed = False
        for kind in ('points', 'models'):
            rows = self._rows(kind)
            existing = {row['id'] for row in rows}
            deleted = set(state['deleted_' + kind])
            for row in plan.get(kind, []):
                if row['id'] in existing or row['id'] in deleted:
                    continue
                if legacy:
                    row['name'] = unique_name((row.get('name') or row['id']).strip(), rows)
                check = new_plan()
                check['settings'] = plan.get('settings', {})
                check[kind] = [row]
                rows.append(validate(check)[kind][0])
                existing.add(row['id'])
                changed = True
            plan[kind] = rows
        # Deleting a shared resource also removes its steps in stale browser drafts.
        plan['steps'] = [s for s in plan['steps'] if not (
            s.get('type') == 'point' and s.get('point') in state['deleted_points'] or
            s.get('type') == 'vla' and s.get('model') in state['deleted_models'])]
        plan.pop('resources', None)
        if legacy:
            repair_vla_names(plan)
        plan = validate(plan)
        if changed:
            for kind in ('points', 'models'):
                self._write(kind, plan[kind])
            self._touch()
        return plan

    def resolve(self, plan):
        with self.transaction():
            return self._resolve(plan)

    def save_resource(self, kind, row, settings=None):
        check = new_plan()
        check['settings'] = settings or {}
        check[kind] = [row]
        row = validate(check)[kind][0]
        with self.transaction():
            state = self._state()
            if row['id'] in state['deleted_' + kind]:
                raise ValueError('此共享资源已被删除；请复制为新资源再保存')
            rows = self._rows(kind)
            index = next((i for i, old in enumerate(rows) if old['id'] == row['id']), len(rows))
            if index == len(rows):
                rows.append(row)
            else:
                rows[index] = row
            validate_names(rows, '共享关键点' if kind == 'points' else 'VLA 模型')
            self._write(kind, rows)
            self._touch(state)
        return row

    def delete_resource(self, kind, identifier):
        field, step_type = ('point', 'point') if kind == 'points' else ('model', 'vla')
        with self.transaction():
            state = self._state()
            if identifier not in state['deleted_' + kind]:
                state['deleted_' + kind].append(identifier)
            # Commit the tombstone first so an interrupted deletion cannot resurrect
            # the resource from another browser's draft or an older saved document.
            self._touch(state)
            self._write(kind, [row for row in self._rows(kind) if row['id'] != identifier])
            for path, document in self._documents():
                steps = [s for s in document['steps'] if not (s['type'] == step_type and s.get(field) == identifier)]
                if len(steps) != len(document['steps']):
                    document['steps'] = steps
                    atomic_json(path, self._reference_document(document))

    def save(self, plan, identifier=None):
        import uuid
        identifier = identifier or uuid.uuid4().hex
        path = self.path(identifier)
        with self.transaction():
            plan = self._resolve(plan)
            atomic_json(path, self._reference_document(plan))
        return identifier

    def load(self, identifier):
        with self.transaction():
            return self._resolve(json.loads(self.path(identifier).read_text()), legacy=True)

    def listing(self):
        with self.transaction():
            return [dict(id=path.stem, name=plan['name'], steps=len(plan['steps']))
                    for path, plan in self._documents()]
