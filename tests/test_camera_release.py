"""Camera handoff and diagnostics without opening USB/CAN devices."""

import sys
import tempfile
import threading
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/PiperRecorder'))
from piper_recorder.camera import OrbbecCamera
from piper_recorder.camera_errors import camera_start_error, device_holders
from easy_embodied_deploy.workflow.hardware import DepthCamera, PiperHardware
from easy_embodied_deploy.workflow.runtime import WorkflowController


class CameraReleaseTests(unittest.TestCase):
    def test_error_identifies_camera_and_visible_holder_without_full_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            device = root / 'sys/bus/usb/devices/2-7.2'
            device.mkdir(parents=True)
            for name, value in [('serial', 'ABC'), ('busnum', '2'), ('devnum', '6')]:
                (device / name).write_text(value)
            process = root / 'proc/123'
            (process / 'fd').mkdir(parents=True)
            (process / 'fd/9').symlink_to('/dev/bus/usb/002/006')
            (process / 'cmdline').write_bytes(b'python\0/home/py/Workflow.py\0--secret\0value\0')
            error = RuntimeError('uvc_open failed: [Path: 2-7.2-6.0, Return Code: -6]')
            holders = device_holders('ABC', str(error), sysfs=root/'sys', proc=root/'proc')
            self.assertEqual(holders, ['PID 123 (python Workflow.py)'])
            with patch('piper_recorder.camera_errors.device_holders', return_value=holders):
                message = str(camera_start_error('agentview', 'ABC', error))
            for text in ('agentview', 'ABC', 'PID 123', '断开设备', 'Return Code: -6'):
                self.assertIn(text, message)
            self.assertNotIn('secret', message)

    def test_nonbusy_error_is_not_misreported_as_workflow_ownership(self):
        with patch('piper_recorder.camera_errors.device_holders', return_value=[]):
            message = str(camera_start_error('right_eye_in_hand', 'ABC', RuntimeError('Return Code: -3')))
        self.assertIn('权限不足', message)
        self.assertNotIn('接口忙', message)

    def test_rgb_and_depth_failed_start_release_pipeline(self):
        for kind in ('rgb', 'depth'):
            with self.subTest(kind=kind):
                pipeline = Mock()
                pipeline.start.side_effect = RuntimeError('uvc_open failed: Return Code: -6')
                sdk = SimpleNamespace(Pipeline=Mock(return_value=pipeline), Config=Mock(),
                    OBFormat=SimpleNamespace(RGB=1),
                    OBSensorType=SimpleNamespace(COLOR_SENSOR=1, DEPTH_SENSOR=2),
                    OBFrameAggregateOutputMode=SimpleNamespace(FULL_FRAME_REQUIRE=1),
                    AlignFilter=Mock(), OBStreamType=SimpleNamespace(COLOR_STREAM=1))
                with patch.dict(sys.modules, pyorbbecsdk=sdk), \
                     patch('easy_embodied_deploy.workflow.hardware.sdk_intrinsics', return_value={}), \
                     patch('piper_recorder.camera_errors.device_holders', return_value=[]):
                    with self.assertRaisesRegex(RuntimeError, 'agentview.*ABC.*接口忙'):
                        if kind == 'rgb':
                            camera = OrbbecCamera('agentview', 'ABC', Mock(), width=640, height=480,
                                                  fps=30, image_size=(640, 480), max_age_s=.5)
                            camera.start(Mock())
                        else:
                            DepthCamera('agentview', 'ABC', Mock())
                pipeline.stop.assert_called_once()
                if kind == 'rgb':
                    self.assertIsNone(camera._pipeline)

    def test_depth_close_releases_handles_and_is_repeatable(self):
        camera = DepthCamera.__new__(DepthCamera)
        camera.stop_event = threading.Event()
        camera.pipeline, camera.thread, camera.align = Mock(), Mock(), Mock()
        camera.thread.is_alive.return_value = False
        camera.latest, camera.error = object(), None
        pipeline, thread = camera.pipeline, camera.thread
        camera.close()
        pipeline.stop.assert_called_once()
        thread.join.assert_called_once()
        self.assertIsNone(camera.pipeline)
        self.assertIsNone(camera.align)
        self.assertIsNone(camera.latest)
        camera.close()

    def test_disconnect_releases_cameras_before_arms_and_lease_and_blocks_late_preview(self):
        events = []
        rig = PiperHardware.__new__(PiperHardware)
        rig.lock, rig.resources = threading.RLock(), ExitStack()
        rig.closed = rig.teaching = rig.emergency = False
        rig.stop = Mock()
        rig.cameras, rig.context = {'agentview': object()}, object()
        rig.resources.callback(lambda: events.append(('lease', rig.context)))
        rig.resources.callback(lambda: events.append(('arm', len(rig.cameras))))
        rig.resources.callback(rig._clear_camera_references)
        rig.resources.callback(lambda: events.append(('camera', True)))
        controller = WorkflowController()
        controller.hardware = rig
        try:
            controller.disconnect()
            self.assertIsNone(controller.hardware)
            self.assertEqual(events, [('camera', True), ('arm', 0), ('lease', None)])
            with self.assertRaisesRegex(RuntimeError, '已断开'):
                rig.camera('agentview')
            rig.close()
            self.assertEqual(len(events), 3)
        finally:
            controller.close()

    def test_stop_error_does_not_skip_other_resource_cleanup(self):
        rig = PiperHardware.__new__(PiperHardware)
        rig.lock, rig.resources = threading.RLock(), ExitStack()
        rig.closed = rig.teaching = rig.emergency = False
        rig.stop = Mock(side_effect=RuntimeError('stale arm feedback'))
        camera, arm = Mock(), Mock()
        rig.resources.callback(arm.disconnect)
        rig.resources.callback(camera.close)
        with self.assertRaisesRegex(RuntimeError, 'stale arm'):
            rig.close()
        camera.close.assert_called_once()
        arm.disconnect.assert_called_once()

    def test_depth_stop_failure_is_reported_and_sdk_references_are_dropped(self):
        camera = DepthCamera.__new__(DepthCamera)
        camera.stop_event = threading.Event()
        camera.pipeline, camera.thread, camera.align = Mock(), Mock(), Mock()
        camera.thread.is_alive.return_value = False
        camera.pipeline.stop.side_effect = RuntimeError('USB stop failed')
        camera.latest, camera.error = object(), None
        with self.assertRaisesRegex(RuntimeError, 'USB stop failed'):
            camera.close()
        self.assertIsNone(camera.pipeline)
        self.assertIsNone(camera.align)
        camera.close()


if __name__ == '__main__':
    unittest.main()
