"""Editing, unique names, GPU selection and last-reference release in real Chromium."""

import asyncio
import os
import tempfile
import threading
from unittest.mock import patch

from playwright.async_api import async_playwright
from easy_embodied_deploy.workflow.console import WorkflowServer


async def main():
    inventory = [dict(index=i, uuid=f'GPU-test-{i}', name=f'Test GPU {i}',
                      memory_used_mb=1024, memory_total_mb=24000) for i in (0, 1)]
    with tempfile.TemporaryDirectory() as directory, patch(
            'easy_embodied_deploy.workflow.console.gpu_status', return_value=inventory):
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
                await page.wait_for_function('gpus.length === 2')
                assert await page.locator('.tab-index').all_text_contents() == ['2', '3', '4']
                assert await page.locator('#point-count').inner_text() == '已有关键点：0'
                await page.locator('#connection-mode').select_option('simulation')
                await page.locator('#connect').click()
                await page.wait_for_function('live.connected')
                await page.locator('#capture-name').fill('基准点')
                await page.locator('#capture').click()
                await page.wait_for_function('plan.points.length === 1')
                await page.locator('#capture-name').fill(' 基准点 ')
                await page.locator('#capture').click()
                await page.wait_for_function("document.getElementById('inline-error').textContent.includes('已存在')")
                assert len(server.store.snapshot()['points']) == 1
                await page.locator('#point-tools summary').click()
                await page.locator('#manual-point').click()
                await page.locator('#point-name').fill('基准点')
                await page.locator('#point-form button[type="submit"]').click()
                await page.locator('#point-error').wait_for(state='visible')
                assert '已存在' in await page.locator('#point-error').inner_text()
                await page.locator('#point-name').fill('另一个点')
                await page.locator('#point-form button[type="submit"]').click()
                await page.wait_for_function('plan.points.length === 2')
                original = page.locator('#points [data-point]').first
                await original.locator('summary').click()
                await original.locator('[data-point-action="copy"]').click()
                await page.wait_for_function('plan.points.length === 3')
                await original.locator('[data-point-action="copy"]').click()
                await page.wait_for_function('plan.points.length === 4')
                assert await page.evaluate('plan.points.map(p=>p.name)') == ['基准点', '另一个点', '基准点 副本', '基准点 副本 2']

                await page.locator('[data-tab="flow"]').click()
                async def create_model(name):
                    await page.locator('#new-model-step').click()
                    await page.locator('[data-model-field="backend"]').select_option('mock')
                    await page.locator('[data-model-field="name"]').fill(name)
                    await page.locator('#model-form button[type="submit"]').click()
                await create_model('取件模型')
                await page.wait_for_function('plan.models.length === 1')
                await create_model(' 取件模型 ')
                await page.locator('#model-error').wait_for(state='visible')
                assert '已存在' in await page.locator('#model-error').inner_text()
                await page.locator('[data-model-field="name"]').fill('放件模型')
                await page.locator('#model-form button[type="submit"]').click()
                await page.wait_for_function('plan.models.length === 2')
                first, second = await page.evaluate('plan.models.map(m=>m.id)')
                await page.locator('#add-model-select').select_option(first)
                await page.locator('#add-vla').click()
                await page.locator('#steps .step').first.locator('[data-step-action="copy"]').click()
                names = await page.evaluate('plan.steps.map(s=>s.name)')
                assert len(names) == len(set(names)) == 4, names
                name_input = page.locator('#steps .step').nth(1).locator('[data-field="name"]')
                old_name = await name_input.input_value()
                await name_input.fill('取件模型')
                await name_input.press('Tab')
                assert await name_input.input_value() == old_name
                await page.wait_for_function("document.getElementById('inline-error').textContent.includes('已存在')")

                # Click at the left edge, then type: digits must append, even on a
                # second click while the number input already has keyboard focus.
                duration = page.locator('[data-field="value"]').first
                await duration.fill('12')
                await duration.press('Tab')
                for digit, expected in [('3', '123'), ('4', '1234')]:
                    await duration.click(position={'x': 9, 'y': 15})
                    await page.evaluate('new Promise(requestAnimationFrame)')
                    await page.keyboard.type(digit)
                    assert await duration.input_value() == expected
                await duration.press('Tab')
                assert await page.evaluate('plan.steps[0].value') == 1234
                await page.locator('[data-field="until"]').first.select_option('actions')
                await duration.fill('12')
                await duration.press('Tab')
                await duration.click(position={'x': 9, 'y': 15})
                await page.evaluate('new Promise(requestAnimationFrame)')
                await page.keyboard.type('3')
                assert await duration.input_value() == '123'
                await duration.press('Tab')
                await page.locator('#save').click()
                await page.wait_for_function('savedId !== null && !dirty')
                await page.reload()
                await page.wait_for_function('plan?.steps.length === 4')
                assert await page.locator('#step-count').inner_text() == '已有步骤：4'

                await page.locator('[data-tab="models"]').click()
                first_row = page.locator(f'#model-list [data-model="{first}"]')
                second_row = page.locator(f'#model-list [data-model="{second}"]')
                await first_row.locator('[data-model-device]').select_option('cuda:1')
                await page.wait_for_function("plan.models[0].device === 'cuda:1'")
                assert server.store.snapshot()['models'][0]['device'] == 'cuda:1'
                await first_row.locator('[data-model-device]').select_option('cuda:0')
                await page.wait_for_function("plan.models[0].device === 'cuda:0'")
                await first_row.locator('[data-model-action="load"]').click()
                await second_row.locator('[data-model-action="load"]').click()
                await page.wait_for_function("live.models.length === 2 && live.models.every(m=>m.stage === 'ready')")
                first_process = server.controller.models.get(first)
                second_process = server.controller.models.get(second)
                await first_row.locator('[data-model-action="load"]').click()
                assert server.controller.models.get(first) is first_process
                await first_row.locator('[data-model-device]').select_option('cpu')
                await page.wait_for_function("plan.models[0].device === 'cpu'")
                assert '需重新加载' in await first_row.locator('[data-model-state]').inner_text()
                await first_row.locator('[data-model-action="load"]').click()
                await page.wait_for_function("(id)=>live.models.some(m=>m.id===id && m.stage==='ready' && m.config.device==='cpu')", arg=first)
                assert first_process.process.poll() is not None
                assert server.controller.models.get(second) is second_process
                await page.screenshot(path='/tmp/workflow-model-devices.png', full_page=True)

                await page.locator('[data-tab="flow"]').click()
                # Two remaining references preserve the worker; the final removal
                # releases only that model, not the other resident model/config.
                while await page.evaluate('(id)=>plan.steps.filter(s=>s.model===id).length', first):
                    index = await page.evaluate('(id)=>plan.steps.findIndex(s=>s.model===id)', first)
                    count = await page.evaluate('plan.steps.length')
                    await page.locator('#steps .step').nth(index).locator('[data-step-action="delete"]').click()
                    await page.wait_for_function('(n)=>plan.steps.length===n-1', arg=count)
                    remaining = await page.evaluate('(id)=>plan.steps.some(s=>s.model===id)', first)
                    assert (first in server.controller.models.models) == remaining
                assert server.controller.models.get(second) is second_process
                assert len(server.store.snapshot()['models']) == 2
                await page.locator('[data-field="until"]').select_option('manual')
                await page.locator('#run').click()
                await page.wait_for_function("live.busy && live.operation === 'workflow' && live.step_type === 'vla'")
                await page.locator('[data-step-action="delete"]').click()
                await page.wait_for_function('plan.steps.length === 0 && live.pending_unloads.length === 1')
                assert second_process.process.poll() is None
                await page.locator('#stop').click()
                await page.wait_for_function('!live.busy && live.models.length === 0 && live.pending_unloads.length === 0')
                assert second_process.process.poll() is not None
                assert len(server.store.snapshot()['models']) == 2
                await page.set_viewport_size({'width': 390, 'height': 844})
                assert await page.evaluate('document.documentElement.scrollWidth <= innerWidth')
                assert not errors, errors
                await browser.close()
            print('Editing: names/copies/reuse, duration and actions caret, GPU routing UI, worker reuse/reload, final-reference/deferred release, navigation counts and mobile layout PASS')
        finally:
            server.controller.close()
            server.shutdown()
            server.server_close()
            worker.join()


if __name__ == '__main__':
    asyncio.run(main())
