"""Real pointer drag preview/cancel and joint-interface persistence; no devices."""

import asyncio
import os
import tempfile
import threading

from playwright.async_api import async_playwright
from easy_embodied_deploy.workflow.console import WorkflowServer


async def main():
    with tempfile.TemporaryDirectory() as directory:
        server = WorkflowServer(('127.0.0.1', 0), directory)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            async with async_playwright() as playwright:
                browser = await playwright.chromium.launch(
                    headless=True, executable_path=os.environ.get('FUSION_TEST_BROWSER'))
                page = await browser.new_page(viewport={'width': 1440, 'height': 1800})
                errors = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                await page.goto(f'http://127.0.0.1:{server.server_port}/')
                await page.locator('#file-tools summary').click()
                await page.locator('#example').click()
                await page.wait_for_function('plan.steps.length === 3')
                original = await page.evaluate('plan.steps.map(step => step.id)')

                async def drag_start(source, destination, after):
                    handle = await page.locator('.step-number').nth(source).bounding_box()
                    target = await page.locator('#steps .step').nth(destination).bounding_box()
                    assert handle and target
                    await page.mouse.move(handle['x']+8, handle['y']+8)
                    await page.mouse.down()
                    await page.mouse.move(handle['x']+12, handle['y']+18)
                    await page.mouse.move(target['x']+30, target['y']+target['height']*(.85 if after else .1), steps=12)
                    await page.locator('.step-placeholder').wait_for(state='visible')
                    # Cards have already moved while the button remains down.
                    target = await page.locator('#steps .step').nth(destination).bounding_box()
                    await page.mouse.move(target['x']+30, target['y']+target['height']*(.85 if after else .1), steps=4)

                # The second card must move before releasing the first card.
                second_before = await page.locator('#steps .step').nth(1).bounding_box()
                await drag_start(0, 2, True)
                assert await page.evaluate('plan.steps.map(step => step.id)') == original
                second_after = await page.locator('#steps .step').nth(1).bounding_box()
                assert second_after['y'] < second_before['y']
                assert await page.evaluate("document.querySelector('#steps').lastElementChild.classList.contains('step-placeholder')")
                await page.screenshot(path='/tmp/workflow-drag-preview.png', full_page=True)
                await page.mouse.up()
                await page.wait_for_function('!document.querySelector(".step-placeholder")')
                reordered = original[1:] + original[:1]
                assert await page.evaluate('plan.steps.map(step => step.id)') == reordered

                # Cancel restores DOM and data; no accidental insertion.
                await drag_start(2, 0, False)
                await page.keyboard.press('Escape')
                await page.mouse.up()
                await page.wait_for_function('!document.querySelector(".step-placeholder")')
                assert await page.evaluate('plan.steps.map(step => step.id)') == reordered
                assert await page.locator('.drag-source').count() == 0

                # Moving the last card to the first slot restores the original.
                await drag_start(2, 0, False)
                await page.mouse.up()
                await page.wait_for_function('!document.querySelector(".step-placeholder")')
                assert await page.evaluate('plan.steps.map(step => step.id)') == original

                await page.locator('#settings').evaluate('(node) => node.closest("details").open = true')
                await page.locator('[data-setting="joint_command"]').select_option('move_j')
                assert await page.evaluate('plan.settings.joint_command') == 'move_j'
                await page.locator('#save').click()
                await page.wait_for_function('savedId !== null && !dirty')
                await page.reload()
                await page.wait_for_function("plan?.settings?.joint_command === 'move_j'")
                assert await page.locator('[data-setting="joint_command"]').input_value() == 'move_j'
                assert not errors, errors
                await browser.close()
            print('Workflow drag preview before release, move both directions, Escape rollback, interface save/reload PASS')
        finally:
            server.controller.close()
            server.shutdown()
            server.server_close()
            worker.join()


if __name__ == '__main__':
    asyncio.run(main())
