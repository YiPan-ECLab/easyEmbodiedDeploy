"""Exercise both deletion UIs and per-file timing approval in Chromium."""

import asyncio
import os
import tempfile
import threading
from pathlib import Path

from playwright.async_api import async_playwright
from test_data_pipeline import make_episode, processor_app
from piper_viewer.server import ViewerServer


async def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        good = make_episode(root / "raw")
        gap = make_episode(root / "raw", gap=True)
        bad = root / "raw/broken.hdf5"
        bad.write_bytes(b"broken")
        servers = [ViewerServer(("127.0.0.1", 0), root / "raw"),
                   processor_app.Server(("127.0.0.1", 0), processor_app.Handler)]
        threads = [threading.Thread(target=s.serve_forever, daemon=True) for s in servers]
        for thread in threads:
            thread.start()
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True, executable_path=os.environ.get("FUSION_TEST_BROWSER"))
                page = await browser.new_page(viewport={"width": 1500, "height": 1050})
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.on("dialog", lambda dialog: dialog.accept())
                await page.goto(f"http://127.0.0.1:{servers[0].server_port}/")
                await page.wait_for_function("state.fusion !== null")
                await page.evaluate("path => selectFile(path)", str(good))
                await page.wait_for_function("[...document.querySelectorAll('#camera-grid img')].every(i => i.complete)")
                await page.locator("#delete-episode").click()
                await page.get_by_role("button", name="撤销删除").wait_for()
                assert not good.exists()
                await page.get_by_role("button", name="撤销删除").click()
                await page.wait_for_function("path => state.episode?.path === path", arg=str(good))
                assert good.exists()

                await page.goto(f"http://127.0.0.1:{servers[1].server_port}/")
                await page.wait_for_function("ui.defaults !== null")
                await page.locator("#inputDir").fill(str(root / "raw"))
                await page.locator("#outputRoot").fill(str(root / "output"))
                await page.locator("#targetFps").fill("30")
                await page.locator('input[name="gapPolicy"][value="reject"]').check()
                await page.locator("#inspectButton").click()
                await page.wait_for_function("ui.plan?.invalid_files === 2")
                assert await page.locator("#fileTable tr").count() == 3
                assert await page.get_by_role("button", name="忽略时序错误", exact=True).count() == 1
                await page.get_by_role("button", name="忽略时序错误", exact=True).click()
                await page.wait_for_function("ui.plan?.invalid_files === 1")
                assert "--allow-timing" in await page.locator("#commandPreview").inner_text()
                await page.locator("#fileTable tr").filter(has_text="broken.hdf5").get_by_role("button", name="删除轨迹").click()
                await page.wait_for_function("ui.plan?.invalid_files === 0 && ui.plan?.episodes === 2")
                assert not bad.exists()
                await page.locator("#deleteResult button").click()
                await page.wait_for_function("ui.plan?.invalid_files === 1 && ui.plan?.episodes === 3")
                assert bad.exists()
                # Any configuration edit invalidates earlier consent.
                await page.locator("#task").fill("New task")
                assert await page.evaluate("Object.keys(ui.timingOverrides).length === 0")
                await page.locator("#inspectButton").click()
                await page.wait_for_function("ui.plan?.invalid_files === 2")
                await page.locator("#issuesOnly").check()
                assert await page.locator("#fileTable tr").count() == 2
                assert await page.evaluate("""() => {
                  const reviewed = ui.plan;
                  renderStatus({status: 'complete', job_id: ui.lastJob,
                    plan: {...reviewed, files: [], episodes: 0}, events: []});
                  return ui.plan === reviewed;
                }""")
                async def delayed(route):
                    await asyncio.sleep(0.3)
                    await route.continue_()
                await page.route("**/api/inspect", delayed)
                await page.locator("#inspectButton").click()
                await page.locator("#task").fill("Changed while checking")
                await page.wait_for_timeout(650)
                assert await page.evaluate("ui.plan === null")
                assert await page.locator("#inspectButton").is_enabled()
                await page.unroute("**/api/inspect", delayed)
                await page.locator("#inspectButton").click()
                await page.wait_for_function("ui.plan?.invalid_files === 2")
                await page.locator("#inspectionPanel").screenshot(path="/tmp/trajectory-review-ui.png")
                assert not errors, errors
                await browser.close()
            print("Trajectory review browser: viewer delete/restore, all-file errors, scoped approval, delete/restore, reset consent PASS")
        finally:
            for server in servers:
                server.shutdown()
                server.server_close()
            for thread in threads:
                thread.join()


if __name__ == "__main__":
    asyncio.run(main())
