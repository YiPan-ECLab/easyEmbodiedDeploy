"""Exercise real conversion and analysis jobs through the unified HTTP proxy.

Run in thirdParty/.venvs/lerobot-pi05/bin/python; uses only temporary fixtures.
"""

import argparse
import json
import tempfile
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from test_data_pipeline import make_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8088")
    args = parser.parse_args()

    def request(module, endpoint, body=None):
        url = f"{args.url}/modules/{module}/api/{endpoint}"
        req = Request(
            url,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(req, timeout=120) as response:
                return json.load(response)
        except HTTPError as exc:
            raise RuntimeError(f"{module}/{endpoint}: {exc.read().decode()}") from exc

    def done(module):
        end = time.monotonic() + 120
        while time.monotonic() < end:
            status = request(module, "status")
            if status["status"] != "running":
                return status
            time.sleep(0.15)
        raise TimeoutError(module)

    # Only launch web consoles; the service start endpoint never moves hardware.
    for module in (
        "viewer",
        "processor",
        "analysis",
        "lerobot-trainer",
        "openpi-trainer",
        "recorder",
    ):
        request_body = json.dumps({"module": module}).encode()
        with urlopen(
            Request(
                args.url + "/api/services/start",
                data=request_body,
                headers={"Content-Type": "application/json"},
            )
        ) as response:
            json.load(response)
    until = time.monotonic() + 60
    while time.monotonic() < until:
        with urlopen(args.url + "/api/status") as response:
            modules = json.load(response)["modules"]
        if all(
            m["state"] == "ready"
            for m in modules
            if m["id"]
            in (
                "viewer",
                "processor",
                "analysis",
                "lerobot-trainer",
                "openpi-trainer",
                "recorder",
            )
        ):
            break
        time.sleep(0.2)
    else:
        raise TimeoutError("Web consoles did not become ready")
    with tempfile.TemporaryDirectory(prefix="easy-http-smoke-") as temp:
        root = Path(temp)
        source = root / "hdf5"
        episode = make_episode(source)
        config = dict(
            input_dir=str(source),
            output_root=str(root / "converted"),
            task="test task",
            target_fps=10,
            source_fps=30,
            action_mode="delta_joint",
            repo_id="local/http-test",
            limit=0,
            gap_policy="reject",
        )
        files = request("viewer", "files?" + urlencode({"directory": str(source)}))
        assert files["count"] == 1
        viewed = request("viewer", "episode?" + urlencode({"path": str(episode)}))
        assert viewed["frames"] == 27
        request(
            "analysis",
            "start",
            {
                "kind": "hdf5",
                "path": str(source),
                "expected_fps": 30,
                "gap_factor": 1.5,
            },
        )
        analyzed = done("analysis")
        assert analyzed["status"] == "complete", analyzed
        assert analyzed["report"]["status"] == "PASS", analyzed
        request("processor", "start", config)
        converted = done("processor")
        assert converted["status"] == "complete", converted
        assert converted["result"]["validation"]["valid"], converted
        request(
            "analysis", "start", {"kind": "lerobot", "path": str(root / "converted")}
        )
        analyzed = done("analysis")
        assert analyzed["status"] == "complete", analyzed
        assert analyzed["report"]["status"] == "PASS", analyzed
        # Validate the same data through both native training console processes.
        for module in ("lerobot-trainer", "openpi-trainer"):
            defaults = request(module, "bootstrap")["defaults"]
            defaults.update(
                dataset_root=str(root / "converted"),
                output_dir=str(root / module),
                device="cpu",
                action_mode="delta_joint",
                wandb_enabled=False,
            )
            report = request(module, "validate", {"config": defaults})
            assert report["dataset"]["frames"] > 0, report
        # API origin protection must survive the reverse proxy.
        try:
            urlopen(
                Request(
                    args.url + "/modules/recorder/api/start",
                    data=b"{}",
                    headers={
                        "Origin": "https://unrelated.invalid",
                        "Content-Type": "application/json",
                    },
                )
            )
        except HTTPError as exc:
            assert exc.code == 403
        else:
            raise AssertionError("cross-origin control accepted")
        print(
            "PASS: viewer, HDF5 analysis, conversion, LeRobot analysis, both training validation APIs, origin guard"
        )


if __name__ == "__main__":
    main()
