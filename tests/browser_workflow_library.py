"""Two browser sessions share persisted points/models and workflow references."""

import asyncio
import json
import os
import tempfile
import threading
from pathlib import Path

from playwright.async_api import async_playwright
from easy_embodied_deploy.workflow.console import WorkflowServer


async def main():
    with tempfile.TemporaryDirectory() as directory:
        server = WorkflowServer(('127.0.0.1', 0), directory)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True, executable_path=os.environ.get('FUSION_TEST_BROWSER'))
                a, b = await browser.new_page(), await browser.new_page()
                errors = []
                for page in (a, b):
                    page.on('pageerror', lambda error: errors.append(str(error)))
                    page.on('dialog', lambda dialog: dialog.accept())
                    await page.goto(f'http://127.0.0.1:{server.server_port}/')
                    await page.wait_for_function('plan !== undefined')
                await a.locator('#point-tools summary').click()
                await a.locator('#manual-point').click()
                await a.locator('#point-name').fill('跨流程基准点')
                await a.locator('#point-form button[type="submit"]').click()
                await a.wait_for_function('plan.points.length === 1')
                await a.locator('[data-tab="models"]').click()
                await a.locator('#add-model').click()
                await a.locator('[data-model-field="name"]').fill('共享取件模型')
                await a.locator('[data-model-field="checkpoint"]').fill('/tmp/shared checkpoint')
                await a.locator('#model-form button[type="submit"]').click()
                await a.wait_for_function('plan.models.length === 1')
                # Neither a workflow save nor a browser-local draft is needed in b.
                await b.wait_for_function('plan.points.length === 1 && plan.models.length === 1')
                await b.reload()
                await b.wait_for_function('plan?.models.length === 1')
                point, model = await a.evaluate('[plan.points[0].id, plan.models[0].id]')
                ids = []
                for page, name in ((a, '流程 A'), (b, '流程 B')):
                    await page.locator('#plan-name').fill(name)
                    await page.locator('[data-tab="flow"]').click()
                    await page.locator('#add-point-step').click()
                    await page.locator('#add-vla').click()
                    await page.locator('#save').click()
                    await page.wait_for_function('savedId !== null && !dirty')
                    ids.append(await page.evaluate('savedId'))
                await a.locator('#file-tools summary').click()
                await a.locator('#new').click()
                await a.wait_for_function('plan.steps.length === 0')
                assert await a.evaluate('[plan.points.length, plan.models.length]') == [1, 1]
                await a.locator('[data-tab="points"]').click()
                assert await a.locator('#points [data-point]').count() == 1
                await a.locator('[data-point-action="edit"]').click()
                await a.locator('#point-name').fill('更新的共享基准点')
                await a.locator('#point-form button[type="submit"]').click()
                await a.wait_for_function("plan.points[0].name === '更新的共享基准点'")
                await a.locator('[data-tab="models"]').click()
                await a.locator('[data-model-action="edit"]').click()
                await a.locator('[data-model-field="checkpoint"]').fill('/tmp/new shared checkpoint')
                await a.locator('#model-form button[type="submit"]').click()
                await b.wait_for_function("plan.models[0].checkpoint === '/tmp/new shared checkpoint'")
                for identifier in ids:
                    loaded = server.store.load(identifier)
                    assert loaded['points'][0]['name'] == '更新的共享基准点'
                    assert loaded['models'][0]['checkpoint'] == '/tmp/new shared checkpoint'
                    raw = json.loads((Path(directory) / f'{identifier}.json').read_text())
                    assert raw['points'] == [] and raw['models'] == []
                    assert raw['steps'][0]['point'] == point and raw['steps'][1]['model'] == model
                await a.locator('#model-list summary').click()
                await a.locator('[data-model-action="delete"]').click()
                await b.wait_for_function('plan.models.length === 0 && plan.steps.length === 1')
                assert all(len(server.store.load(identifier)['steps']) == 1 for identifier in ids)
                await a.screenshot(path='/tmp/workflow-shared-library.png', full_page=True)
                assert not errors, errors
                await browser.close()
            print('Shared library: immediate persistence, two browsers, new flow, global edits, references, deletion PASS')
        finally:
            server.controller.close()
            server.shutdown()
            server.server_close()
            worker.join()


if __name__ == '__main__':
    asyncio.run(main())
