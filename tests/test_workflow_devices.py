"""GPU routing, independent workers and deferred release, without GPU allocation."""

import copy
import os
import unittest
from unittest.mock import patch

from easy_embodied_deploy.workflow.devices import normalize_device, worker_launch
from easy_embodied_deploy.workflow.models import ModelPool
from easy_embodied_deploy.workflow.runtime import WorkflowController
from easy_embodied_deploy.workflow.plan import new_plan
from test_workflow import wait_for


class DeviceTests(unittest.TestCase):
    def test_device_validation(self):
        self.assertEqual(normalize_device(' cuda '), 'cuda:0')
        self.assertEqual(normalize_device('CUDA:02'), 'cuda:2')
        for value in ('cuda:-1', 'cuda:0,1', '', None, 'gpu:0'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_device(value)

    def test_physical_gpu_binding_is_the_same_for_both_backends(self):
        with patch('easy_embodied_deploy.workflow.devices.gpu_status', return_value=[dict(index=2, uuid='GPU-physical-2')]), \
             patch.dict(os.environ, CUDA_VISIBLE_DEVICES='5,4', JAX_PLATFORMS='cpu', XLA_PYTHON_CLIENT_PREALLOCATE='true'):
            for backend in ('lerobot', 'openpi'):
                config = dict(id='m', backend=backend, device='cuda:2')
                worker, env = worker_launch(config)
                self.assertEqual(config['device'], 'cuda:2')
                self.assertEqual(worker['device'], 'cuda:0')
                self.assertEqual(worker['requested_device'], 'cuda:2')
                self.assertEqual(env['CUDA_VISIBLE_DEVICES'], 'GPU-physical-2')
                self.assertEqual(env['JAX_PLATFORMS'], 'cuda')
                self.assertEqual(env['XLA_PYTHON_CLIENT_PREALLOCATE'], 'false')
                cpu, env = worker_launch(dict(config, device='cpu'))
                self.assertEqual(cpu['device'], 'cpu')
                self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '')
                self.assertEqual(env['JAX_PLATFORMS'], 'cpu')
                with self.assertRaisesRegex(ValueError, 'GPU 1'):
                    worker_launch(dict(config, device='cuda:1'))
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '5,4')

    def test_same_device_workers_coexist_reuse_move_and_fail_independently(self):
        pool = ModelPool()
        self.addCleanup(pool.close)
        models = [dict(id=i, name=i, backend='mock', action_mode='absolute_joint', device='cuda:0') for i in ('a', 'b')]
        pool.preload(models)
        wait_for(lambda: all(m['stage'] == 'ready' for m in pool.snapshot()))
        first, second = pool.get('a'), pool.get('b')
        pool.preload(models)
        self.assertIs(pool.get('a'), first)
        self.assertIs(pool.get('b'), second)
        models[0]['device'] = 'cuda:1'
        pool.preload(models)
        wait_for(lambda: pool.get('a').stage == 'ready')
        self.assertIsNot(pool.get('a'), first)
        self.assertIsNotNone(first.process.poll())
        self.assertIs(pool.get('b'), second)
        pool.preload([dict(models[0], fail_load=True)])
        wait_for(lambda: pool.get('a').stage == 'failed')
        self.assertEqual(second.stage, 'ready')
        pool.preload([models[0]])
        wait_for(lambda: pool.get('a').stage == 'ready')
        pool.close('a')
        self.assertEqual([m['id'] for m in pool.snapshot()], ['b'])

    def test_invalid_gpu_does_not_unload_existing_models(self):
        pool = ModelPool()
        self.addCleanup(pool.close)
        model = dict(id='a', name='a', backend='mock', action_mode='absolute_joint')
        pool.preload([model])
        original = pool.get('a')
        with patch('easy_embodied_deploy.workflow.devices.gpu_status', return_value=[]):
            with self.assertRaises(ValueError):
                pool.preload([dict(model, backend='openpi', device='cuda:999')])
        self.assertIs(pool.get('a'), original)
        self.assertIsNone(original.process.poll())

    def test_last_reference_release_waits_for_active_run(self):
        controller = WorkflowController()
        self.addCleanup(controller.close)
        plan = new_plan()
        plan['models'] = [dict(id='m', name='模拟', backend='mock', action_mode='absolute_joint')]
        plan['steps'] = [dict(id='s', name='模拟控制', type='vla', model='m', until='manual')]
        controller.connect(True)
        controller.start(copy.deepcopy(plan), home_before=False)
        wait_for(lambda: any(m['stage'] == 'ready' for m in controller.models.snapshot()))
        process = controller.models.get('m')
        controller.release_model('m')
        self.assertEqual(controller.snapshot()['pending_unloads'], ['m'])
        self.assertIsNone(process.process.poll())
        controller.stop()
        wait_for(lambda: not controller.models.snapshot())
        self.assertIsNotNone(process.process.poll())
        self.assertEqual(controller.snapshot()['pending_unloads'], [])


if __name__ == '__main__':
    unittest.main()
