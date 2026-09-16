"""Read-only console interaction and responsive-layout regression.

Run against a test workbench. Never starts training, hardware or deployment.
"""

import argparse
import asyncio
from pathlib import Path

from playwright.async_api import async_playwright


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8098")
    parser.add_argument("--browser")
    parser.add_argument("--screenshots", type=Path, default=Path("/tmp/easy-theme-browser"))
    args = parser.parse_args()
    args.screenshots.mkdir(parents=True, exist_ok=True)
    errors = []
    async with async_playwright() as p:
        options = {"headless": True}
        if args.browser:
            options["executable_path"] = args.browser
        browser = await p.chromium.launch(**options)
        page = await browser.new_page(viewport={"width": 1440, "height": 1024})
        page.on("pageerror", lambda e: errors.append(str(e)))
        modules = (await (await page.request.get(args.url + "/api/status")).json())["modules"]
        for module in modules:
            name = module["id"]
            response = await page.request.post(args.url + "/api/services/start", data={"module": name})
            assert response.ok, await response.text()
            # Start is asynchronous; a cold console may still return the proxy's
            # temporary 503 page. Wait as the workbench does before navigating.
            for _ in range(150):
                response = await page.request.get(args.url + "/api/status")
                assert response.ok, await response.text()
                status = await response.json()
                current = next(item for item in status["modules"] if item["id"] == name)
                assert current["state"] != "failed", current
                if current["state"] == "ready":
                    break
                await asyncio.sleep(0.2)
            else:
                raise TimeoutError(f"{name} console did not become ready: {current}")
            response = await page.goto(f"{args.url}/modules/{name}/")
            assert response is not None and response.ok, name
            await page.wait_for_function("() => document.body.innerText.length > 80")
            await page.wait_for_timeout(1000)
            if name.endswith("trainer"):
                bootstrap = await (await page.request.get(f"{args.url}/modules/{name}/api/bootstrap")).json()
                datasets = [d for d in bootstrap["datasets"] if d.get("action_mode") == "delta_joint"]
                assert datasets and bootstrap["models"], "This integration check needs an installed dataset and base model"
                dataset = datasets[0]
                await page.locator("#dataset_root").fill(dataset["path"])
                await page.locator("#pretrained").fill(bootstrap["models"][0]["path"])
                await page.locator('[data-profile="full"]').click()
                assert await page.locator("#finetune_mode").input_value() == "full"
                await page.locator('[data-profile="lora"]').click()
                assert await page.locator("#finetune_mode").input_value() == "lora"
                await page.locator("#previewButton").click()
                await page.wait_for_function("label => document.getElementById('commandPreview').textContent.includes(label)", arg=dataset["label"])
            if name == "calibration":
                assert await page.locator("#apply-intrinsics").is_disabled()
                assert "打开相机" in await page.locator("#intrinsics-context").inner_text()
                await page.locator("#job").select_option("global_right")
                assert "T_right_base_global_camera" in await page.locator("#transform-direction").inner_text()
                await page.locator("#job").select_option("left_wrist")
                assert "T_left_flange_left_camera" in await page.locator("#transform-direction").inner_text()
            # Expanded controls must remain operable on narrow screens too.
            await page.locator("details").evaluate_all("nodes => nodes.forEach(n => n.open = true)")
            for width in (1440, 768, 390):
                await page.set_viewport_size({"width": width, "height": 1024})
                await page.wait_for_timeout(100)
                layout = await page.evaluate("""() => ({
                    width: innerWidth, scroll: document.documentElement.scrollWidth,
                    scheme: getComputedStyle(document.documentElement).colorScheme
                })""")
                assert layout["scroll"] <= width + 1, (name, layout)
                assert "dark" not in layout["scheme"], (name, layout)
                if width in (1440, 390):
                    await page.screenshot(path=str(args.screenshots / f"{name}-{width}.png"))
            print(f"{name}: expanded controls / desktop / tablet / phone PASS", flush=True)
        assert not errors, errors
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
