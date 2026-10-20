"""Exercise the maintenance UI using fake CAN feedback, never real devices."""
import asyncio
import os
import sys
import tempfile
import threading
from pathlib import Path

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tests')]
from easy_embodied_deploy.workflow.console import WorkflowServer
from easy_embodied_deploy.workflow.hardware import MockHardware
from easy_embodied_deploy.workflow.recovery import inspect_arm, restore_arm
from test_workflow_recovery import Arm, LIMITS


class FakeHardware(MockHardware):
    simulated = False
    limits = LIMITS
    def __init__(self):
        super().__init__()
        self.arms = {'left': Arm(True), 'right': Arm()}
    def diagnose(self, side):
        return dict(inspect_arm(self.arms[side], self.limits), side=side)
    def restore(self, side, check, **kwargs):
        return dict(restore_arm(self.arms[side], self.limits, check, **kwargs), side=side)


async def main():
    with tempfile.TemporaryDirectory() as directory:
        server = WorkflowServer(('127.0.0.1', 0), directory)
        hardware = FakeHardware()
        server.controller.hardware = hardware
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True, executable_path=os.environ.get('FUSION_TEST_BROWSER'))
                page = await browser.new_page(viewport={'width': 1360, 'height': 1000})
                errors = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                await page.goto(f'http://127.0.0.1:{server.server_port}/')
                await page.locator('#arm-diagnose').click()
                await page.wait_for_function('armDiagnosis?.eligible === true')
                assert '0x00' in await page.locator('#arm-diagnosis').inner_text()
                assert not hardware.arms['right'].calls
                await page.locator('#arm-restore').click()
                await page.wait_for_function('live.recovery_result?.recovered === true')
                assert hardware.arms['right'].enabled
                assert hardware.arms['left'].calls == []
                await page.locator('#arm-side').select_option('left')
                await page.wait_for_function('armDiagnosis?.arm_status === 1')
                assert await page.locator('#arm-restore').is_disabled()
                await page.locator('#arm-support').check()
                assert await page.locator('#arm-restore').is_enabled()
                hardware.arms['left'].fault = True
                await page.locator('#arm-refresh').click()
                await page.wait_for_function('armDiagnosis?.eligible === false')
                assert await page.locator('#arm-restore').is_disabled()
                assert '堵转' in await page.locator('#arm-diagnosis').inner_text()
                assert not errors, errors
                await browser.close()
        finally:
            server.shutdown(); server.server_close(); server.controller.close()
    print('Workflow recovery browser: diagnosis, side isolation, restore, fault rejection PASS')


asyncio.run(main())
