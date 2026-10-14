"""Isolated calibration UI regression: temporary data, synthetic feedback, no hardware.

Run with a Python environment containing OpenCV, NumPy and Playwright:
  python tests/browser_calibration_validation.py --browser /path/to/chrome
"""

import argparse
import asyncio
import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import numpy as np
from playwright.async_api import async_playwright

from test_resources_calibration import (
    CAMERA, CalibrationHandler, IndependentValidationTests, synthetic_samples,
    flange_transform, base_transform, inverse, board_pose,
)
from easy_embodied_deploy.pose_actions import matrix_rpy


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--browser')
    args = parser.parse_args()
    fixture = IndependentValidationTests()
    fixture.setUp()
    server = None
    try:
        left, held = fixture.session('left_wrist')
        right, _ = fixture.session('right_wrist')
        gl, _ = fixture.session('global_left', flange_transform([.3, .1, .2, .1, -.2, .1]))
        gr, _ = fixture.session('global_right', flange_transform([-.4, .2, .3, -.2, .1, .3]))
        relation = base_transform(gl['result']['matrix'], gr['result']['matrix'])
        board_left = flange_transform([.4, .1, .8, .1, -.3, .2])
        observations = []
        for record, board in ((left, board_left), (right, inverse(relation) @ board_left)):
            sample = synthetic_samples('wrist', 24)[0][-1]
            target, _ = board_pose(sample['objects'], sample['pixels'], CAMERA)
            sample['T_base_flange'] = (board @ inverse(target) @ inverse(record['result']['matrix'])).tolist()
            updated = fixture.add_validation(record, sample, 'dual_validation', '固定板-1')
            observations.append(updated['validation_samples'][-1]['id'])

        rgb = np.zeros((480, 640, 3), np.uint8)
        matrix = np.asarray(held[0]['T_base_flange'])
        pose = np.r_[matrix[:3, 3], matrix_rpy(matrix[:3, :3])].tolist()
        camera = dict(left['camera'], pose_side='left')

        class Rig:
            opened = False

            def status(self):
                return dict(open=self.opened, camera=camera if self.opened else None)

            def open(self, *_):
                self.opened = True
                return camera

            def close(self):
                self.opened = False

            def read(self):
                return rgb, {}, camera

            def capture(self, with_pose=False):
                frames = [(rgb.copy(), dict(
                    camera_received_monotonic_s=10+i*.2, sample_monotonic_s=10+i*.2,
                    sample_wall_time_s=100+i*.2, pose_sdk_timestamp_s=100+i*.2,
                    pose=pose if with_pose else None, image_age_s=.01, pose_age_s=.01,
                    image_pose_skew_s=0)) for i in range(3)]
                return rgb, dict(frames[-1][1]), camera, frames

        server = ThreadingHTTPServer(('127.0.0.1', 0), CalibrationHandler)
        server.store, server.rig, server.operations = fixture.store, Rig(), threading.RLock()
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        detection = (np.asarray(held[0]['objects']), np.asarray(held[0]['pixels']),
                     list(range(len(held[0]['pixels']))), rgb)
        with mock.patch('easy_embodied_deploy.calibration.detect', return_value=detection):
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True, **(
                    {'executable_path': args.browser} if args.browser else {}))
                page = await browser.new_page(viewport={'width': 1440, 'height': 1000})
                errors = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                base = f'http://127.0.0.1:{server.server_port}/'
                await page.goto(base)
                await page.locator(f'#sessions option[value="{left["id"]}"]').wait_for(state='attached')
                await page.locator('#sessions').select_option(left['id'])
                await page.wait_for_function("document.getElementById('sample-count').textContent==='20 个样本'")
                await page.locator('#open').click()
                await page.wait_for_function("document.getElementById('rig-status').textContent.includes('已打开')")
                await page.locator('#preview-live').uncheck()
                await page.locator('#sample-purpose').select_option('validation')
                await page.locator('#still').check()
                await page.locator('#capture').click()
                await page.wait_for_function("document.getElementById('validation-summary').textContent.includes('1 个独立验证样本')")
                assert '均在参考阈值内' in await page.locator('#validation-summary').inner_text()
                assert '20 个样本' == await page.locator('#sample-count').inner_text()
                assert '法兰变化' in await page.locator('#capture-quality').inner_text()
                current = fixture.store.load(left['id'])
                assert current['result'] == left['result']
                row = page.locator('#validation-list tr').filter(has_text='独立验证')
                await row.get_by_role('button', name='查看', exact=True).click()
                await page.wait_for_function("document.getElementById('preview').complete && document.getElementById('preview').naturalWidth===640")
                await row.get_by_role('button', name='移除', exact=True).click()
                await page.wait_for_function("document.getElementById('validation-summary').textContent.includes('尚未完成')")
                assert fixture.store.load(left['id'])['result'] == left['result']

                for record in (left, right, gl, gr):
                    await page.locator(f'#export-list input[value="{record["id"]}"]').check()
                await page.get_by_text('双臂统一到左基座的一致性检查', exact=True).click()
                await page.locator('#load-dual').click()
                await page.locator(f'#dual-left option[value="{observations[0]}"]').wait_for(state='attached')
                await page.locator('#dual-left').select_option(observations[0])
                await page.locator('#dual-right').select_option(observations[1])
                await page.locator('#dual-fixed').check()
                await page.locator('#compare-bases').click()
                await page.wait_for_function("document.getElementById('dual-report').textContent.includes('within_thresholds')")
                assert json.loads(await page.locator('#dual-report').text_content())['within_thresholds']
                await page.locator('#export').click()
                await page.locator('#download-export').wait_for(state='visible')
                response = await page.request.get(base + await page.locator('#download-export').get_attribute('href'))
                bundle = await response.json()
                assert response.ok
                assert bundle['derived_transforms']['T_left_base_right_base']['independently_validated']
                assert len(bundle['dual_base_validation']) == 1
                assert len(bundle['calibrations']) == 4
                await page.set_viewport_size({'width': 390, 'height': 844})
                assert await page.evaluate('document.documentElement.scrollWidth <= innerWidth')
                await page.screenshot(path='/tmp/calibration-validation-mobile.png', full_page=True)
                await page.set_viewport_size({'width': 1440, 'height': 1000})
                await page.screenshot(path='/tmp/calibration-validation-desktop.png', full_page=True)
                assert not errors, errors
                await browser.close()
                print('PASS: independent capture/view/remove, unchanged solution, dual check, export, mobile; no hardware')
    finally:
        if server:
            server.shutdown()
            server.server_close()
            thread.join()
        fixture.doCleanups()


if __name__ == '__main__':
    asyncio.run(main())
