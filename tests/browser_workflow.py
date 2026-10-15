"""Exercise the workflow UI with simulated hardware and real model IPC."""

import asyncio
import json
import os
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np
from playwright.async_api import async_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from easy_embodied_deploy.workflow.console import WorkflowServer


async def main():
    with tempfile.TemporaryDirectory() as directory:
        server=WorkflowServer(('127.0.0.1',0),directory)
        worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
        try:
            async with async_playwright() as p:
                browser=await p.chromium.launch(headless=True,executable_path=os.environ.get('FUSION_TEST_BROWSER'))
                page=await browser.new_page(viewport={'width':1440,'height':1100})
                errors=[]
                page.on('pageerror',lambda error:errors.append(str(error)))
                page.on('dialog',lambda dialog:dialog.accept())
                await page.goto(f'http://127.0.0.1:{server.server_port}/')
                await page.locator('#file-tools summary').click()
                await page.locator('#example').click()
                await page.locator('#connection-mode').select_option('simulation')
                await page.wait_for_function('plan.steps.length === 3')
                await page.locator('#connect').click()
                await page.wait_for_function('live.connected')
                await page.locator('#run').click()
                await page.wait_for_function("live.state === 'completed' || live.state === 'failed'",timeout=20000)
                assert await page.evaluate("live.state === 'completed'"),await page.locator('#run-detail').inner_text()
                assert '已就绪' in await page.locator('#model-status').inner_text()
                await page.locator('.tabs .tool-menu summary').click()
                await page.locator('[data-tab="models"]').click()
                await page.locator('#model-editor details summary').click()
                for key,value in dict(policy_hz='12',control_hz='80',smoothing_time_constant='0',
                                      rtc_horizon='12',rtc_queue_threshold='30').items():
                    await page.locator(f'[data-model-field="{key}"]').fill(value)
                    await page.locator(f'[data-model-field="{key}"]').press('Tab')
                await page.locator('[data-model-field="backend"]').select_option('openpi')
                await page.locator('#model-editor details summary').click()
                for key,value in dict(sample_steps='7',n_action_steps='9',rtc_max_guidance_weight='3').items():
                    await page.locator(f'[data-model-field="{key}"]').fill(value)
                    await page.locator(f'[data-model-field="{key}"]').press('Tab')
                await page.locator('[data-model-field="rtc_schedule"]').select_option('linear')
                await page.locator('[data-model-field="rtc_mask_grippers"]').uncheck()
                await page.locator('[data-model-field="backend"]').select_option('mock')
                await page.locator('.tabs .tool-menu summary').click()
                await page.locator('[data-tab="points"]').click()
                original_arms = await page.evaluate('JSON.stringify(plan.points[0].arms)')
                await page.locator('[data-point-action="edit"]').first.click()
                await page.locator('#point-name').fill('双臂公共基准点')
                await page.locator('#point-notes').fill('第 1、3 步共同引用')
                await page.locator('#point-form button[type="submit"]').click()
                await page.wait_for_function("plan.points[0].name === '双臂公共基准点'")
                assert await page.evaluate('JSON.stringify(plan.points[0].arms)') == original_arms
                await page.locator('[data-point-action="add"]').first.click()
                assert await page.evaluate('plan.steps.length') == 4
                await page.locator('#save').click()
                await page.wait_for_function('savedId !== null')
                await page.reload()
                await page.wait_for_function('plan?.steps.length === 4')
                actual=await page.evaluate('plan.models[0]')
                for key,value in dict(policy_hz=12,control_hz=80,smoothing_time_constant=0,
                                      rtc_horizon=12,rtc_queue_threshold=30,sample_steps=7,
                                      n_action_steps=9,rtc_max_guidance_weight=3,
                                      rtc_schedule='linear',rtc_mask_grippers=False).items():
                    assert actual[key]==value,(key,actual)
                assert await page.evaluate("plan.steps.filter(s=>s.point===plan.points[0].id).length") == 3
                await page.locator('[data-tab="points"]').click()
                await page.locator('#capture-name').fill('新的拖动点')
                await page.locator('#teach').click()
                await page.wait_for_function('live.teaching')
                await page.locator('#capture').click()
                await page.wait_for_function('plan.points.length === 2')
                assert 'J1' in await page.locator('#points').inner_text()
                assert 'J6' in await page.locator('#points').inner_text()
                recorded_id = await page.evaluate('plan.points[1].id')
                await page.locator(f'[data-point="{recorded_id}"] [data-point-action="recapture"]').click()
                assert await page.evaluate('plan.points.length') == 2
                assert await page.evaluate('plan.points[1].id') == recorded_id
                assert await page.locator('#run').is_enabled()
                await page.locator('#save').click()
                await page.locator('#run').click()
                await page.wait_for_function("!live.busy && live.state === 'completed'", timeout=25000)
                assert not server.controller.hardware.teaching
                assert any('自动平滑回零' in e['message'] for e in server.controller.events)
                await page.locator('#point-tools summary').click()
                await page.locator('#import-library').click()
                await page.locator('#library-dialog').wait_for(state='visible')
                assert await page.locator('[data-library-import]').count() >= 2
                await page.locator('#close-library').click()

                # Depth selection uses an exported calibration bundle and does not move.
                rgb,depth,camera=server.controller.hardware.camera('agentview')
                matrix=np.eye(4)
                calibration=dict(schema='easy_embodied_camera_calibration',calibrations=[dict(
                    job='global_left',camera=camera,mount_id='sim',result=dict(
                        transform='T_left_base_global_camera',matrix=matrix.tolist(),accepted=True))])
                calibration_path=Path(directory)/'simulation_calibration.json'
                calibration_path.write_text(json.dumps(calibration))
                await page.locator('.tabs .tool-menu summary').click()
                await page.locator('[data-tab="camera"]').click()
                await page.locator('#calibration').fill(str(calibration_path))
                await page.locator('#freeze').click()
                await page.wait_for_function('frozen !== null')
                await page.locator('#camera-image').click(position={'x':200,'y':150})
                before=len(server.controller.hardware.commands)
                await page.locator('#save-pick').click()
                await page.wait_for_function('plan.points.length === 3')
                assert len(server.controller.hardware.commands) == before
                assert await page.evaluate("plan.points.at(-1).space === 'pose'")
                await page.locator('.point-more summary').last.click()
                await page.locator('[data-point-action="copy"]').last.click()
                await page.wait_for_function('plan.points.length === 4')
                await page.locator('.point-more summary').last.click()
                await page.locator('[data-point-action="delete"]').last.click()
                await page.wait_for_function('plan.points.length === 3')
                assert await page.evaluate('plan.steps.length') == 4
                # Homing is separate from emergency and leaves the point library intact.
                before_points=await page.evaluate('JSON.stringify(plan.points)')
                await page.locator('#teach').click()
                await page.wait_for_function('live.teaching')
                assert await page.locator('#run').is_enabled()
                await page.locator('#home').click()
                await page.wait_for_function("live.state === 'completed' && live.operation === 'home' && !live.busy")
                assert await page.evaluate('JSON.stringify(plan.points)') == before_points
                assert not server.controller.hardware.teaching
                assert all(v == 0 for v in server.controller.hardware.joints[:6])
                assert server.controller.hardware.joints[6] == .04
                await page.locator('#device-tools summary').click()
                await page.locator('#telemetry').click()
                await page.wait_for_function("document.getElementById('feedback').textContent.includes('J1–J6')")
                await page.locator('#recover').click()
                await page.wait_for_function('!live.teaching')
                await page.locator('#device-tools summary').click()
                await page.screenshot(path='/tmp/workflow-desktop.png',full_page=True)
                await page.set_viewport_size({'width':390,'height':844})
                assert await page.evaluate('document.documentElement.scrollWidth <= innerWidth')
                await page.screenshot(path='/tmp/workflow-mobile.png',full_page=True)
                # A new page served by an old connected backend must not expose
                # the incompatible teaching controls as usable.
                await page.route('**/api/bootstrap', lambda route: route.fulfill(
                    status=200,content_type='application/json',body=json.dumps(dict(
                        plan=server.controller.plan,points=[],workflows=[],calibrations=[],serials={},
                        state=server.controller.snapshot()))))
                await page.reload()
                await page.locator('#version-notice').wait_for(state='visible')
                assert await page.locator('#teach').is_disabled()
                assert await page.locator('#home').is_disabled()
                assert not await page.locator('#emergency').is_disabled()
                assert not errors,errors
                await browser.close()
            print('Workflow browser: multi-stage run, shared point reuse/edit, save/reload, teach/capture, depth pick, mobile PASS')
        finally:
            server.controller.close();server.shutdown();server.server_close();worker.join()


if __name__=='__main__':
    asyncio.run(main())
