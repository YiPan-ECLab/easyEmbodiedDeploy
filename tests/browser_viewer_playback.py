"""Real-browser playback, bounded prefetch, looping and stale-request checks."""

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
        first = make_episode(root / "first", frames=180, task_frames=180,
                             last_active=179, image_size=32)
        second = make_episode(root / "second", image_size=32)
        server = ViewerServer(("127.0.0.1", 0), root)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True, executable_path=os.environ.get("FUSION_TEST_BROWSER"))
                page = await browser.new_page()
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                await page.goto(f"http://127.0.0.1:{server.server_port}/")
                await page.wait_for_function("state.files.length > 0 && state.fusion !== null")
                await page.evaluate("path => selectFile(path)", str(first))
                assert await page.evaluate("state.episode.frames === 180")
                await page.evaluate("""() => {
                  window.displayed = []; window.misaligned = [];
                  const original = updateFrame;
                  updateFrame = (...args) => {
                    original(...args);
                    if (!state.playing) return;
                    displayed.push(state.frame);
                    for (const img of document.querySelectorAll('#camera-grid img')) {
                      if (Number(new URL(img.src).searchParams.get('index')) !== state.frame)
                        misaligned.push(img.src);
                    }
                  };
                  togglePlayback();
                }""")
                await page.wait_for_function("state.frame >= 35")
                assert await page.evaluate("state.imageFrames.size <= 18 && misaligned.length === 0")
                await page.locator('#speed-control button[data-speed="2"]').click()
                await page.wait_for_function("state.frame >= 65")
                await page.evaluate("stopPlayback()")
                paused = await page.evaluate("state.frame")
                await page.wait_for_timeout(200)
                assert await page.evaluate("state.frame") == paused

                # Lookahead wraps around the end without requesting invalid indices.
                await page.evaluate("updateFrame(178, true); togglePlayback()")
                await page.wait_for_function("state.frame < 10")
                await page.evaluate("stopPlayback()")

                # A delayed camera response may not overwrite a later manual seek.
                async def delayed(route):
                    await asyncio.sleep(0.2)
                    await route.continue_()
                await page.route("**/api/frame?**", delayed)
                await page.evaluate("state.imageFrames.clear(); updateFrame(10, true); togglePlayback(); stopPlayback(); updateFrame(120, true)")
                await page.wait_for_timeout(600)
                assert await page.evaluate("state.frame === 120 && !state.playing && state.prefetchFrames.length === 0")
                await page.evaluate("togglePlayback()")
                await page.evaluate("path => selectFile(path)", str(second))
                await page.wait_for_timeout(600)
                assert await page.evaluate("state.frame === 0 && !state.playing && state.imageFrames.size === 0")
                assert await page.evaluate("""path => [...document.querySelectorAll('#camera-grid img')].every(img =>
                  new URL(img.src).searchParams.get('path') === path && img.complete && img.naturalWidth > 0)""", str(second))
                await page.unroute("**/api/frame?**", delayed)

                # Fusion preview prefetches retained indices, including the loop.
                await page.evaluate("""() => {
                  state.fusionPreview = true;
                  state.fusion.indices = [0, 1, 4, 7, 10, 13, 19, 23];
                  state.fusion.plan.playback_fps = 30;
                  displayed = []; togglePlayback();
                }""")
                await page.wait_for_timeout(650)
                await page.evaluate("stopPlayback()")
                assert await page.evaluate("displayed.length > 8 && displayed.every(i => state.fusion.indices.includes(i))")
                assert await page.evaluate("[...state.imageFrames.keys()].every(i => state.fusion.indices.includes(i)) && misaligned.length === 0")
                assert not errors, errors
                await browser.close()
            print("Viewer browser: playback, speed, loop, bounded cache, seek, file switch, fusion PASS")
        finally:
            server.shutdown()
            server.server_close()
            worker.join()


if __name__ == "__main__":
    asyncio.run(main())
