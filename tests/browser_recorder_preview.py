"""Exercise the real preview HTTP path and UI using synthetic cameras only."""

import asyncio
import os
import threading
import time
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from playwright.async_api import async_playwright

from easy_embodied_deploy.camera_preview import CameraPreview
from easy_embodied_deploy.recorder_console import Handler, RecorderSession


class Camera:
    failed = False

    def __init__(self, color):
        self.rgb = np.full((224, 224, 3), color, dtype=np.uint8)

    def read(self):
        if self.failed:
            raise RuntimeError('camera disconnected')
        return SimpleNamespace(rgb=self.rgb, timestamp=time.monotonic())


async def main():
    config = dict(output='/tmp/test-preview', hz=30, max_task_gap_s=.35,
                  start_joint_delta_rad=.01, start_gripper_delta_m=.002)
    cameras = dict(zip(('agentview', 'left_eye_in_hand', 'right_eye_in_hand'),
                       (Camera([180, 50, 40]), Camera([40, 160, 60]), Camera([40, 80, 190]))))
    preview = CameraPreview(cameras)
    with patch('easy_embodied_deploy.recorder_console.defaults', return_value=config):
        session = RecorderSession()
        session.process = Mock(pid=123)
        session.process.poll.return_value = None
        session.generation = 1
        session.state.update(phase='instruction', token=1, accepts=['submit'], preview_port=preview.port)
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        server.daemon_threads = True
        server.session = session
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True, executable_path=os.environ.get('FUSION_TEST_BROWSER'))
                page = await browser.new_page(viewport={'width':1440, 'height':1100})
                errors, requests, posts = [], [], []
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.on('request', lambda request: requests.append(request.url) if '/api/camera-frame?' in request.url else None)
                async def guard(route):
                    if route.request.method != 'GET':
                        posts.append(route.request.url)
                        await route.abort()
                    else:
                        await route.continue_()
                await page.route('**/api/**', guard)
                await page.goto(f'http://127.0.0.1:{server.server_port}/')
                await page.wait_for_function("document.querySelectorAll('.camera-tile.live').length === 3")
                assert await page.locator('.camera-panel img').evaluate_all('images => images.every(i => i.naturalWidth === 224 && !i.hidden)')
                await page.screenshot(path='/tmp/recorder-preview-desktop.png')
                cameras['left_eye_in_hand'].failed = True
                await page.wait_for_function("!document.querySelector('[data-camera=left_eye_in_hand]').classList.contains('live')")
                assert await page.locator('.camera-tile.live').count() == 2
                cameras['left_eye_in_hand'].failed = False
                await page.wait_for_function("document.querySelectorAll('.camera-tile.live').length === 3")
                # Hiding the panel suspends requests instead of accumulating images.
                await page.evaluate('previewVisible = false; updateCameraPreview()')
                await page.wait_for_timeout(300)
                count = len(requests)
                await page.wait_for_timeout(400)
                assert len(requests) == count
                await page.evaluate('previewVisible = true; updateCameraPreview()')
                await page.wait_for_function("document.querySelectorAll('.camera-tile.live').length === 3")
                session.process.poll.return_value = 0
                await page.wait_for_function("document.querySelectorAll('.camera-tile.live').length === 0")
                session.generation += 1
                session.process.poll.return_value = None
                await page.wait_for_function("document.querySelectorAll('.camera-tile.live').length === 3")
                await page.set_viewport_size({'width':390, 'height':844})
                await page.wait_for_timeout(200)
                assert await page.evaluate('document.documentElement.scrollWidth <= innerWidth')
                await page.screenshot(path='/tmp/recorder-preview-mobile.png', full_page=True)
                assert not errors, errors
                assert not posts, posts
                session.process.stdin.write.assert_not_called()
                await browser.close()
            print('Recorder preview: three live cameras, failure/recovery, visibility, reconnect, mobile PASS')
        finally:
            server.shutdown()
            server.server_close()
            worker.join()
            preview.close()


if __name__ == '__main__':
    asyncio.run(main())
