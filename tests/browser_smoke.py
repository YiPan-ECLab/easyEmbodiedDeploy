"""Browser integration; starts web services only, never hardware or training.

Run: uv run --no-project --with playwright python tests/browser_smoke.py
"""

import argparse
import asyncio
from pathlib import Path

from playwright.async_api import async_playwright


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8088")
    parser.add_argument("--browser", help="Optional installed Chromium executable")
    parser.add_argument(
        "--screenshots", type=Path, default=Path("/tmp/easy-embodied-browser")
    )
    args = parser.parse_args()
    args.screenshots.mkdir(parents=True, exist_ok=True)
    errors = []
    async with async_playwright() as p:
        options = {"headless": True}
        if args.browser:
            options["executable_path"] = args.browser
        browser = await p.chromium.launch(**options)
        page = await browser.new_page(viewport={"width": 1440, "height": 1024})
        page.on("pageerror", lambda error: errors.append(str(error)))
        await page.goto(args.url)
        await page.wait_for_selector(".module-card")
        assert await page.locator(".module-card").count() == 10
        await page.screenshot(
            path=str(args.screenshots / "workbench.png"), full_page=True
        )
        modules = (await (await page.request.get(args.url + "/api/status")).json())[
            "modules"
        ]
        for module in modules:
            await page.locator(f'.nav-item[data-module="{module["id"]}"]').click()
            locator = page.locator(f"#frame-{module['id']}")
            await locator.wait_for(timeout=60000)
            frame = await (await locator.element_handle()).content_frame()
            await frame.wait_for_load_state("domcontentloaded")
            await frame.wait_for_function("document.body.innerText.length > 80")
            await page.wait_for_timeout(900)
            assert await frame.locator("script[src]").count() > 0
            print("MODULE", module["id"], "PASS", flush=True)
        await page.locator('.nav-item[data-module="recorder"]').click()
        await page.screenshot(
            path=str(args.screenshots / "recorder.png"), full_page=True
        )
        await page.set_viewport_size({"width": 390, "height": 844})
        await page.screenshot(path=str(args.screenshots / "mobile.png"), full_page=True)
        # Exercise the real JS against a deterministic record.py operator protocol.
        pedal = await browser.new_page()
        pedal.on("pageerror", lambda error: errors.append(str(error)))
        defaults = (
            await (
                await pedal.request.get(args.url + "/modules/recorder/api/bootstrap")
            ).json()
        )["defaults"]
        state = dict(
            phase="instruction",
            token=1,
            session_id=1,
            running=True,
            pending=False,
            stopping=False,
            frames=0,
            accepts=["submit"],
            saved=[],
            logs=[],
            config=defaults,
        )
        inputs = []

        async def protocol(route):
            if route.request.url.endswith("/input"):
                message = route.request.post_data_json
                assert message["token"] == state["token"]
                inputs.append(message)
                old = state["phase"]
                state["phase"] = {
                    "instruction": "waiting_motion",
                    "waiting_motion": "review",
                    "review": "instruction",
                }[old]
                state["token"] += 1
                state["accepts"] = (
                    ["finish"] if state["phase"] == "waiting_motion" else ["submit"]
                )
            data = (
                {"defaults": defaults, "status": state}
                if route.request.url.endswith("/bootstrap")
                else state
            )
            await route.fulfill(json=data)

        await pedal.route("**/modules/recorder/api/**", protocol)
        await pedal.goto(args.url + "/modules/recorder/")
        await pedal.wait_for_function(
            "document.getElementById('pedal').disabled === false"
        )
        await pedal.locator("#instruction").fill("test instruction")
        await pedal.keyboard.down("Enter")
        await pedal.wait_for_function(
            "document.getElementById('phase-tag').textContent === '等待动作'"
        )
        await pedal.keyboard.down("Enter")  # repeat while held across the new phase
        await pedal.wait_for_timeout(650)
        assert len(inputs) == 1, inputs
        await pedal.keyboard.up("Enter")
        await pedal.keyboard.press("Enter")
        await pedal.wait_for_function(
            "document.getElementById('phase-tag').textContent === '待保存'"
        )
        await pedal.keyboard.press("Enter")
        await pedal.wait_for_function(
            "document.getElementById('phase-tag').textContent === '等待指令'"
        )
        assert [item["action"] for item in inputs] == ["submit", "finish", "submit"]
        assert inputs[0]["value"] == "test instruction"
        assert inputs[2]["value"] == ""
        assert await pedal.locator("#instruction").input_value() == ""
        await pedal.keyboard.press("Enter")
        await pedal.wait_for_function(
            "document.getElementById('phase-tag').textContent === '等待动作'"
        )
        await pedal.keyboard.press("Enter")
        await pedal.wait_for_function(
            "document.getElementById('phase-tag').textContent === '待保存'"
        )
        await pedal.locator("#discard").focus()
        await pedal.keyboard.press("Enter")
        await pedal.wait_for_timeout(300)
        assert inputs[-1]["value"] == "n"
        assert not errors, errors
        print("PEDAL repeat/save/discard PASS; browser errors=0", flush=True)
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
