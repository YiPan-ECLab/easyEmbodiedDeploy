"""Exercise task inheritance through the real conversion HTTP/UI, without devices."""

import asyncio
import os
import tempfile
import threading
from pathlib import Path

import h5py
from playwright.async_api import async_playwright

from test_data_pipeline import make_episode, processor_app


async def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        sources = [root / 'left', root / 'right']
        tasks = ['拿起左侧物体', 'Place the right object']
        for source, task in zip(sources, tasks):
            source.mkdir()
            episode = make_episode(source)
            with h5py.File(episode, 'r+') as file:
                file['data/demo_0'].attrs['language'] = task
        server = processor_app.Server(('127.0.0.1', 0), processor_app.Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            async with async_playwright() as playwright:
                browser = await playwright.chromium.launch(
                    headless=True, executable_path=os.environ.get('FUSION_TEST_BROWSER'))
                page = await browser.new_page(viewport={'width': 1440, 'height': 1100})
                errors = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                await page.goto(f'http://127.0.0.1:{server.server_port}/')
                await page.wait_for_function('ui.defaults !== null')
                assert await page.locator('#inheritTask').is_checked()
                assert await page.locator('#task').is_disabled()
                await page.locator('#inputDir').fill('\n'.join(map(str, sources)))
                await page.locator('#outputRoot').fill(str(root / 'output'))
                await page.locator('#inspectButton').click()
                await page.wait_for_function('ui.plan !== null')
                assert await page.evaluate('ui.plan.errors') == []
                assert await page.evaluate('ui.plan.tasks') == tasks
                for task in tasks:
                    assert task in await page.locator('#fileTable').inner_text()
                assert '--task-source hdf5' in await page.locator('#commandPreview').inner_text()
                await page.locator('#inheritTask').uncheck()
                assert await page.locator('#task').is_enabled()
                await page.locator('#task').fill('统一指令')
                await page.locator('#inspectButton').click()
                await page.wait_for_function("ui.plan?.tasks?.[0] === '统一指令'")
                assert await page.evaluate('ui.plan.errors') == []
                assert await page.evaluate('ui.plan.files.map(file => file.task)') == ['统一指令'] * 2
                assert '--task-source override' in await page.locator('#commandPreview').inner_text()
                await page.locator('input[name="actionMode"][value="absolute_base_pose"]').check()
                assert await page.locator('[data-pose-cleaning]').is_visible()
                assert not await page.locator('#poseSingularityFilter').is_checked()
                await page.locator('#poseSingularityFilter').check()
                assert await page.locator('#poseSingularityThreshold').is_enabled()
                await page.locator('#inspectButton').click()
                await page.wait_for_function('ui.plan?.pose_cleaning?.enabled === true')
                assert await page.evaluate('ui.plan.episodes') == 0
                assert '奇异过滤' in await page.locator('#cleaningSummary').inner_text()
                assert '--pose-singularity-filter' in await page.locator('#commandPreview').inner_text()
                await page.locator('input[name="actionMode"][value="absolute_joint"]').check()
                assert not await page.locator('[data-pose-cleaning]').is_visible()
                assert not await page.locator('#cleaningSummary').is_visible()
                assert '--pose-singularity-filter' not in await page.locator('#commandPreview').inner_text()
                assert not errors, errors
                await browser.close()
            print('Processor UI: per-HDF5 task inheritance, multi-directory inspection, explicit override PASS')
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    asyncio.run(main())
