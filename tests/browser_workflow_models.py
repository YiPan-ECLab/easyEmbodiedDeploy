"""Model list/editor/reference lifecycle over HTTP with mock inference, no hardware."""

import asyncio
import copy
import os
import tempfile
import threading
from unittest.mock import patch

from playwright.async_api import async_playwright
from easy_embodied_deploy.workflow.console import WorkflowServer
from easy_embodied_deploy.workflow.models import ModelProcess


class TestModelProcess(ModelProcess):
    """Keep real UI configs and pool matching; only the worker uses fake inference."""
    def __init__(self, config, launch=None):
        super().__init__({**config, 'backend': 'mock'})
        self.config = copy.deepcopy(config)


async def main():
    with tempfile.TemporaryDirectory() as directory, patch(
            'easy_embodied_deploy.workflow.models.ModelProcess', TestModelProcess):
        server = WorkflowServer(('127.0.0.1', 0), directory)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            async with async_playwright() as playwright:
                browser = await playwright.chromium.launch(headless=True,
                    executable_path=os.environ.get('FUSION_TEST_BROWSER'))
                page = await browser.new_page(viewport={'width': 1440, 'height': 1100})
                errors = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.on('dialog', lambda dialog: dialog.accept())
                await page.goto(f'http://127.0.0.1:{server.server_port}/')
                await page.locator('[data-tab="flow"]').click()
                await page.locator('#add-vla').click()
                await page.locator('#close-model').click()
                assert await page.evaluate('plan.models.length') == 0

                async def fill_model(name, path):
                    await page.locator('[data-model-field="name"]').fill(name)
                    await page.locator('[data-model-field="action_mode"]').select_option('absolute_joint')
                    # Submit directly while the last edited input still has focus.
                    await page.locator('[data-model-field="checkpoint"]').fill(path)
                    await page.locator('#model-form button[type="submit"]').click()
                    await page.wait_for_function("!document.getElementById('model-dialog').open")

                await page.locator('#add-vla').click()
                await fill_model('模型一', '/tmp/模型一 checkpoint')
                first = await page.evaluate('plan.models[0].id')
                await page.locator('#new-model-step').click()
                await fill_model('模型二', '/tmp/模型二 checkpoint')
                second = await page.evaluate('plan.models[1].id')
                assert await page.evaluate('plan.steps.map(s => s.model)') == [first, second]
                await page.locator('#add-model-select').select_option(first)
                await page.locator('#add-vla').click()
                assert await page.evaluate('plan.steps.map(s => s.model)') == [first, second, first]
                await page.locator('[data-edit-step-model]').nth(1).click()
                assert await page.locator('[data-model-field="checkpoint"]').input_value() == '/tmp/模型二 checkpoint'
                await page.locator('[data-model-field="checkpoint"]').fill('/tmp/discard-edit')
                await page.locator('#close-model').click()
                assert await page.evaluate('plan.models[1].checkpoint') == '/tmp/模型二 checkpoint'

                await page.locator('[data-tab="models"]').click()
                first_row = page.locator(f'#model-list [data-model="{first}"]')
                await first_row.locator('[data-model-action="edit"]').click()
                await fill_model('取件模型', '/tmp/updated checkpoint')
                assert await page.evaluate('plan.steps.filter(s => s.model === plan.models[0].id).length') == 2
                await first_row.locator('summary').click()
                await first_row.locator('[data-model-action="copy"]').click()
                await fill_model('备用副本', '/tmp/updated checkpoint')
                third = await page.evaluate('plan.models[2].id')
                await page.locator('#model-search').fill('备用副本')
                assert await page.locator('#model-list [data-model]').count() == 1
                await page.locator('#model-list summary').click()
                await page.locator('[data-model-action="delete"]').click()
                assert await page.evaluate('plan.models.length') == 2
                assert await page.evaluate('plan.steps.length') == 3
                await page.locator('#model-search').fill('')
                await first_row.locator('[data-model-action="load"]').click()
                await page.wait_for_function("live.models?.some(m => m.stage === 'ready')", timeout=20000)
                loaded = server.controller.models.get(first)
                assert loaded.config['checkpoint'] == '/tmp/updated checkpoint'
                await page.wait_for_function("document.querySelector('[data-model-state]').textContent === '已就绪'")
                await first_row.locator('[data-model-action="edit"]').click()
                await fill_model('取件模型', '/tmp/new unsaved-to-worker checkpoint')
                assert await first_row.locator('[data-model-state]').inner_text() == '配置已修改 · 需重新加载'
                await page.locator('#restore-loaded').click()
                assert await page.evaluate('plan.models[0].checkpoint') == '/tmp/new unsaved-to-worker checkpoint'
                # A lost browser model entry can be recovered from the running worker.
                await page.evaluate('plan.models = plan.models.slice(1); renderModels()')
                await page.locator('#restore-loaded').click()
                assert await page.evaluate('(id) => plan.models.find(m => m.id === id).checkpoint', first) == '/tmp/new unsaved-to-worker checkpoint'
                await page.evaluate('(id) => { plan.models.sort((a,b) => (b.id === id) - (a.id === id)); changed(); renderModels(); }', first)
                await first_row.locator('[data-model-action="edit"]').click()
                await fill_model('取件模型', '/tmp/updated checkpoint')
                await page.locator('[data-tab="flow"]').click()
                for field in await page.locator('[data-field="until"]').all():
                    await field.select_option('actions')
                for field in await page.locator('[data-field="value"]').all():
                    await field.fill('1')
                    await field.press('Tab')
                await page.locator('#save').click()
                await page.wait_for_function('savedId !== null && !dirty')
                await page.reload()
                await page.wait_for_function('plan?.steps.length === 3')
                assert await page.evaluate('plan.models[0].checkpoint') == '/tmp/updated checkpoint'
                await page.locator('#connection-mode').select_option('simulation')
                await page.locator('#connect').click()
                await page.wait_for_function('live.connected')
                await page.locator('#run').click()
                await page.wait_for_function("live.state === 'completed' || live.state === 'failed'", timeout=25000)
                assert await page.evaluate('live.state') == 'completed', await page.evaluate('live.error')
                assert server.controller.models.get(first) is loaded
                assert server.controller.models.get(second).config['checkpoint'] == '/tmp/模型二 checkpoint'
                await page.locator('[data-tab="models"]').click()
                await first_row.locator('[data-model-action="unload"]').click()
                await page.wait_for_function('live.models.length === 1')
                await page.screenshot(path='/tmp/workflow-model-list.png', full_page=True)
                await page.set_viewport_size({'width': 390, 'height': 844})
                assert await page.evaluate('document.documentElement.scrollWidth <= innerWidth')
                assert not errors, errors
                await browser.close()
            print('Models: create/cancel/edit/copy/delete/search, exact step references, persistence, preload/run reuse, restore loaded config, changed-config badge, unload PASS')
        finally:
            server.controller.close()
            server.shutdown()
            server.server_close()
            worker.join()


if __name__ == '__main__':
    asyncio.run(main())
