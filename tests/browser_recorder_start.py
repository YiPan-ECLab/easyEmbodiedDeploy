"""Recorder start-point UI through real HTTP; synthetic feedback, no devices."""
import asyncio
import os
import tempfile
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from playwright.async_api import async_playwright
from easy_embodied_deploy import recording_start
from easy_embodied_deploy.recorder_console import Handler, RecorderSession
from easy_embodied_deploy.workflow.library import SharedPlanStore


async def main():
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        library=root/'datasets/workflows';library.mkdir(parents=True)
        right=[.1,.2,-.3,0,0,.4,.03]
        left=[-.1,.3,-.4,0,0,-.5,.04]
        shared=SharedPlanStore(library)
        shared.save_resource('points',dict(id='right-start',name='右侧基准',space='joint',arms={'right':right}))
        shared.save_resource('models',dict(id='vla',name='不应出现在采集起点选项中',backend='mock',action_mode='absolute_joint'))
        def defaults():
            return dict(output='/tmp/record-test',hz=30,max_task_gap_s=.35,
                        start_joint_delta_rad=.01,start_gripper_delta_m=.002,
                        start_point=recording_start.load())
        with patch.object(recording_start,'ROOT',root), \
             patch.object(recording_start,'START_PATH',root/'start.json'), \
             patch.object(recording_start,'capture',return_value={'left':left}) as capture, \
             patch('easy_embodied_deploy.recorder_console.defaults',side_effect=defaults):
            server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
            server.daemon_threads=True;server.session=RecorderSession()
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:
                async with async_playwright() as p:
                    browser=await p.chromium.launch(headless=True,executable_path=os.environ.get('FUSION_TEST_BROWSER'))
                    page=await browser.new_page(viewport={'width':1440,'height':1100})
                    errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
                    await page.goto(f'http://127.0.0.1:{server.server_port}/')
                    await page.wait_for_function('configured')
                    await page.locator('#start-point-panel summary').click()
                    assert await page.locator('#start-point-custom').is_hidden()
                    await page.locator('#start-point-mode').select_option('custom')
                    assert len(recording_start.workflow_points()) == 1
                    await page.locator('#start-point-import').click()
                    await page.wait_for_function("startPoint.mode==='custom'")
                    assert await page.evaluate('startPoint.arms.right')==right
                    await page.locator('#start-point-left').click()
                    await page.wait_for_function('startPoint.arms.left[6]===.04')
                    capture.assert_called_once_with('left')
                    await page.locator('#start-point-name').fill('采集基准')
                    await page.locator('#start-point-save').click()
                    await page.wait_for_function("document.getElementById('start-point-message').textContent.includes('已保存')")
                    await page.reload();await page.wait_for_function('configured')
                    assert await page.evaluate('startPoint.name')=='采集基准'
                    assert await page.evaluate('config().start_point.arms')==dict(left=left,right=right)
                    assert '返回起点' in await page.locator('#start').inner_text()
                    await page.locator('#start-point-panel summary').click()
                    assert '左主臂 + 左从臂' in await page.locator('#start-point-values').inner_text()
                    await page.set_viewport_size({'width':390,'height':844})
                    assert await page.evaluate('document.documentElement.scrollWidth <= innerWidth')
                    await page.screenshot(path='/tmp/recorder-start-mobile.png',full_page=True)
                    await page.locator('#start-point-mode').select_option('zero')
                    await page.locator('#start-point-save').click()
                    await page.wait_for_function("document.getElementById('start-point-message').textContent.includes('恢复默认')")
                    assert recording_start.load()['mode']=='zero'
                    assert server.session.process is None
                    assert not errors,errors
                    await browser.close()
                print('Recorder start: import, selected follower capture, four-arm target display, persistence, zero reset, mobile PASS')
            finally:
                server.shutdown();server.server_close();thread.join()


if __name__=='__main__':asyncio.run(main())
