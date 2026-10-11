"""Exercise actual annotations, preview, draft reload, and export in Chromium.

Requires Playwright and the data-pipeline test environment. No hardware access.
"""

import asyncio
import os
import tempfile
import threading
from pathlib import Path

from playwright.async_api import async_playwright
from test_data_pipeline import make_episode
from piper_viewer.server import ViewerServer


async def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        make_episode(root/"raw")
        server = ViewerServer(("127.0.0.1", 0), root/"raw")
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        errors = []
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True, executable_path=os.environ.get("FUSION_TEST_BROWSER"))
                page = await browser.new_page(viewport={"width": 1440, "height": 1100})
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.on("response", lambda response: errors.append(f"HTTP {response.status}: {response.url}")
                        if response.status >= 400 and "/api/" in response.url else None)
                await page.goto(f"http://127.0.0.1:{server.server_port}")
                await page.locator(".file-row").first.click()
                await page.wait_for_function("document.querySelector('#fusion-summary').textContent.includes('24 → 24')")
                await page.locator("#frame-input").fill("4")
                await page.locator("#frame-input").press("Tab")
                await page.locator("#fusion-mark-start").click()
                await page.locator("#frame-input").fill("13")
                await page.locator("#frame-input").press("Tab")
                await page.locator("#fusion-mark-end").click()
                await page.locator("#fusion-add").click()
                await page.wait_for_function("document.querySelector('#fusion-summary').textContent.includes('24 → 18')")
                assert await page.locator("#fusion-strip span").evaluate("el => el.getBoundingClientRect().width > 0 && el.getBoundingClientRect().height > 0")
                await page.locator("#fusion-preview").click()
                await page.wait_for_function("document.querySelector('#fusion-preview').textContent === '返回原始回放'")
                await page.locator("#frame-input").fill("4")
                await page.locator("#frame-input").press("Tab")
                await page.locator("#next-frame").click()
                assert await page.locator("#frame-input").input_value() == "7"
                await page.locator("#previous-frame").click()
                assert await page.locator("#frame-input").input_value() == "4"
                await page.locator("#play-toggle").click()
                await page.wait_for_timeout(150)
                await page.locator("#play-toggle").click()
                await page.reload()
                await page.locator(".file-row").first.click()
                await page.wait_for_function("document.querySelector('#fusion-summary').textContent.includes('24 → 18')")
                await page.locator("#fusion-segments [data-edit]").click()
                await page.locator("#fusion-factor").fill("2")
                await page.locator("#fusion-add").click()
                await page.wait_for_function("document.querySelector('#fusion-summary').textContent.includes('24 → 20')")
                await page.locator("#fusion-export").click()
                await page.get_by_role("button", name="打开导出文件").wait_for(timeout=60000)
                assert len(list((root/"raw_fusion").glob("*.hdf5"))) == 1
                await page.locator(".fusion-panel").screenshot(path="/tmp/easyembodied-fusion-desktop.png")
                await page.get_by_role("button", name="打开导出文件").click()
                await page.wait_for_function("document.querySelector('#fusion-summary').textContent.includes('20 → 20')")
                assert "mixed" in await page.locator("#episode-metrics").inner_text()
                # A first rAF callback can predate performance.now() captured
                # by the play click. It must never produce indices[-1].
                assert await page.evaluate("frameAtMediaTime(-0.001) === state.fusion.indices[0]")
                requests = await page.evaluate("""() => {
                  stopPlayback(); updateFrame(0, true);
                  const original = preloadFrameImages, requested = [];
                  preloadFrameImages = (index) => { requested.push(index); return Promise.resolve([]); };
                  togglePlayback(); cancelAnimationFrame(state.animationFrame);
                  playbackTick(state.playbackStartedAt - 0.5, state.playbackToken);
                  stopPlayback(); preloadFrameImages = original;
                  return requested;
                }""")
                assert requests == [], requests
                await page.locator("#play-toggle").click()
                await page.wait_for_function("state.frame > 0 || !state.playing")
                assert await page.evaluate("state.playing"), await page.locator("#toast").inner_text()
                await page.evaluate("stopPlayback()")
                await page.set_viewport_size({"width": 390, "height": 844})
                await page.locator("#sidebar-toggle").click()
                await page.wait_for_function("document.querySelector('.sidebar').getBoundingClientRect().right <= 1")
                await page.locator(".fusion-panel").scroll_into_view_if_needed()
                await page.screenshot(path="/tmp/easyembodied-fusion-mobile.png")
                assert await page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
                assert not errors, errors
                await browser.close()
            print("Fusion browser: annotation, preview, editing, reload, export, mobile PASS")
        finally:
            server.shutdown()
            server.server_close()
            worker.join()


if __name__ == "__main__":
    asyncio.run(main())
