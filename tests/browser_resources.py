"""Exercise new consoles over HTTP in Chromium; never connect hardware.

Creates and removes only UUID-named test resources. Run against your test
workbench: uv run --no-project --with playwright python tests/browser_resources.py
"""

import argparse
import asyncio
import json
import shutil
import subprocess
import uuid
from pathlib import Path

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8098")
    parser.add_argument("--browser")
    parser.add_argument(
        "--metrics-url", help="Optional read-only console with real diagnostic records"
    )
    parser.add_argument(
        "--screenshots", type=Path, default=Path("/tmp/easy-resources-browser")
    )
    args = parser.parse_args()
    args.screenshots.mkdir(parents=True, exist_ok=True)
    # A known-truth fixture tests solver/export via real HTTP, without a camera.
    prepared = subprocess.run(
        [
            str(ROOT / "thirdParty/.venvs/piper-home/bin/python"),
            "-c",
            """
import sys,json,uuid
from pathlib import Path
sys.path[:0]=['src','tests']
from easy_embodied_deploy.calibration import CalibrationStore
from easy_embodied_deploy.calibration_math import DEFAULT_BOARD
from test_resources_calibration import CAMERA,synthetic_samples
store=CalibrationStore(Path.cwd()/'datasets/calibration')
camera=dict(role='left_eye_in_hand',serial='synthetic-browser-test',sensor='COLOR_SENSOR',width=640,height=480,intrinsics=CAMERA)
s=store.create('left_wrist',camera,DEFAULT_BOARD,'自动测试（合成数据，测试后删除）','synthetic-test-only')
s['samples'],_=synthetic_samples('wrist')
for sample in s['samples']: sample['id']=uuid.uuid4().hex
store.save(s)
print(json.dumps({'id':s['id']}))
""",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    identifier = json.loads(prepared.stdout)["id"]
    cleanup_calibrations = [identifier]
    name = f"_browser_resource_test_{uuid.uuid4().hex}"
    errors = []
    try:
        async with async_playwright() as p:
            options = {"headless": True}
            if args.browser:
                options["executable_path"] = args.browser
            browser = await p.chromium.launch(**options)
            context = await browser.new_context(
                viewport={"width": 1440, "height": 1000},
                permissions=["clipboard-read", "clipboard-write"],
            )

            async def api(route, body=None):
                response = await (
                    context.request.get(args.url + route)
                    if body is None
                    else context.request.post(args.url + route, data=body)
                )
                result = await response.json()
                assert response.ok, result
                return result

            for module in ("files", "calibration"):
                await api("/api/services/start", {"module": module})
            page = await context.new_page()
            page.on("pageerror", lambda error: errors.append(str(error)))
            await page.goto(args.url + "/modules/files/")
            await page.locator('[data-root="datasets"]').click()
            await page.wait_for_function(
                "document.getElementById('current').textContent.endsWith('/datasets')"
            )
            await page.locator("#mkdir").click()
            await page.locator("#dialog-input").fill(name)
            await page.locator("#confirm").click()
            row = page.locator("#items tr").filter(has_text=name)
            await row.get_by_role("button", name="重命名").click()
            await page.locator("#dialog-input").fill(name + "_renamed")
            await page.locator("#confirm").click()
            await row.get_by_role("button", name="复制路径").click()
            await page.wait_for_function(
                "document.getElementById('status').textContent.includes('已复制')"
            )
            await row.get_by_role("button", name="删除", exact=True).click()
            await page.locator("#confirm").click()
            await page.locator("#trash-toggle").click()
            trashed = page.locator("#trash-items tr").filter(has_text=name)
            await trashed.get_by_role("button", name="恢复").click()
            await row.get_by_role("button", name="删除", exact=True).click()
            await page.locator("#confirm").click()
            await trashed.get_by_role("button", name="永久删除").click()
            await page.locator("#dialog-input").fill(name + "_renamed")
            await page.locator("#confirm").click()
            await page.wait_for_function(
                "document.getElementById('status').textContent==='已永久删除'"
            )
            await page.screenshot(
                path=str(args.screenshots / "files.png"), full_page=True
            )
            print("FILES create/rename/copy/trash/restore/purge PASS", flush=True)

            await page.goto(args.url + "/modules/calibration/")
            await page.locator("#sessions option", has_text="自动测试").first.wait_for(
                state="attached"
            )
            assert await page.locator("#columns").input_value() == "4"
            assert await page.locator("#rows").input_value() == "6"
            assert await page.locator("#square-mm").input_value() == "53.5"
            await page.locator("#columns").fill("7")
            await page.locator("#square-mm").fill("24")
            href = await page.locator("#board-download").get_attribute("href")
            response = await context.request.get(
                args.url + "/modules/calibration/" + href
            )
            assert response.ok and 'width="212.0mm"' in await response.text()
            await page.locator("#board-kind").select_option("charuco")
            assert await page.locator("#charuco-options").is_visible()
            await page.locator("#sessions").select_option(identifier)
            await page.wait_for_function(
                "document.getElementById('sample-count').textContent==='20 个样本'"
            )
            await page.locator("#solve").click()
            await page.wait_for_function(
                "document.getElementById('result').textContent.includes('T_left_flange_left_camera')"
            )
            await page.locator(f'#export-list input[value="{identifier}"]').check()
            await page.locator("#export").click()
            await page.locator("#download-export").wait_for(state="visible")
            href = await page.locator("#download-export").get_attribute("href")
            export_id = href.split("id=")[-1]
            cleanup_calibrations.append(export_id)
            response = await context.request.get(
                args.url + "/modules/calibration/" + href
            )
            bundle = await response.json()
            assert bundle["training_integration"] is False
            assert len(bundle["missing_jobs"]) == 3
            await page.screenshot(
                path=str(args.screenshots / "calibration.png"), full_page=True
            )
            await page.set_viewport_size({"width": 390, "height": 844})
            assert await page.evaluate(
                "document.documentElement.scrollWidth <= innerWidth"
            )
            await page.screenshot(
                path=str(args.screenshots / "calibration-mobile.png"), full_page=True
            )
            print(
                "CALIBRATION custom board/known-truth solve/export/mobile PASS",
                flush=True,
            )

            if args.metrics_url:
                await page.set_viewport_size({"width": 1440, "height": 1000})
                await page.goto(args.metrics_url)
                await page.wait_for_function(
                    "document.getElementById('inferenceMedian').textContent.includes('ms')"
                )
                assert (await page.locator("#starvedTicks").inner_text()).isdigit()
                assert "Hz" in await page.locator("#achievedHz").inner_text()
                assert "jsonl" in await page.locator("#diagnosticsPath").inner_text()
                assert not await page.locator("#latencyEmpty").is_visible()
                assert not await page.locator("#queueEmpty").is_visible()
                await page.locator("#inferenceMedian").scroll_into_view_if_needed()
                await page.screenshot(
                    path=str(args.screenshots / "openpi-real-metrics.png")
                )
                print(
                    "OPENPI real recorded inference/starvation/control frequency rendered PASS",
                    flush=True,
                )
            assert not errors, errors
            await browser.close()
    finally:
        for item in cleanup_calibrations:
            assert len(item) == 32 and all(c in "0123456789abcdef" for c in item)
            shutil.rmtree(ROOT / "datasets/calibration" / item, ignore_errors=True)
        # Cleanup only our own uniquely named directory or trash buckets.
        import sys

        sys.path.insert(0, str(ROOT / "src"))
        from easy_embodied_deploy.files import FileStore

        store = FileStore(ROOT, references=lambda: [])
        for suffix in ("", "_renamed"):
            target = ROOT / "datasets" / (name + suffix)
            if target.exists():
                item = store.trash("datasets", target.name)
                store.purge("datasets", item["id"], target.name)
        for item in store.trash_list("datasets"):
            if item["path"] in (name, name + "_renamed"):
                store.purge("datasets", item["id"], item["name"])


if __name__ == "__main__":
    asyncio.run(main())
