"""Resume the frozen E3 experiment from its caches, then run later phases.

Run with the project's Python: python research/jev/resume_e3.py.
Each phase uses the normal jev_research.run entry point and writes its own
auditable artifacts. A failed phase stops the sequence; cached successful VLM
responses are never requested again.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.request


ROOT = Path(__file__).resolve().parents[2]
VARIANT = ROOT / "config" / "variant.json"
sys.path.insert(0, str(ROOT))
from jev_research import config, vlm  # noqa: E402


def log(message):
    print(f"[resume_e3] {message}", flush=True)


def set_phase(original, phase):
    value = dict(original)
    value["phase"] = phase
    with tempfile.NamedTemporaryFile("w", dir=VARIANT.parent, delete=False) as f:
        json.dump(value, f, indent=2)
        f.write("\n")
        tmp = f.name
    os.replace(tmp, VARIANT)


def key_remaining():
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/key",
        headers={"Authorization": "Bearer " + vlm.load_key()},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        data = json.load(response)["data"]
    return data.get("limit_remaining")


def run_phase(original, phase):
    set_phase(original, phase)
    log(f"starting {phase}")
    result = subprocess.run([sys.executable, "-m", "jev_research.run"], cwd=ROOT)
    log(f"{phase} exit={result.returncode}")
    return result.returncode == 0


def artifact(phase):
    return Path(config.RUNS_DIR) / f"{config.TASK}_{phase}_none_seed0" / f"{phase}.json"


def vlm_complete():
    path = artifact("e3vlm")
    if not path.exists():
        return False
    with path.open() as f:
        result = json.load(f)
    return all(result[k]["ok_total"] == result[k]["jobs"]
               for k in ("semantic", "direct", "ablation_2frame"))


def main():
    original = json.loads(VARIANT.read_text())
    try:
        # A post-launch E3 guard edit changed the source fingerprint. Refresh
        # the local E0 audit before asking the API for more annotations.
        try:
            config.require_integrity()
        except RuntimeError:
            log("E0 fingerprint changed; refreshing integrity audit")
            if not run_phase(original, "e0"):
                raise RuntimeError("E0 integrity audit failed")
        config.require_integrity()

        # The model occasionally omits a required JSON field. The production
        # pass fails closed, while its cache keeps valid responses. Relaunching
        # requests only missing/invalid records, with no schema relaxation.
        # Direct JSON has a higher malformed-response rate than the semantic
        # schema. Each attempt requests only the records still missing, so
        # allow enough passes for all six direct chunks to clear the gate.
        if vlm_complete():
            log("complete VLM artifact found; no further OpenRouter calls")
        else:
            max_attempts = 16
            for attempt in range(1, max_attempts + 1):
                remaining = key_remaining()
                log(f"VLM attempt {attempt}; key limit remaining=${remaining:.2f}")
                if remaining is not None and remaining < 1:
                    raise RuntimeError("OpenRouter key has less than $1 remaining")
                if run_phase(original, "e3vlm") and vlm_complete():
                    break
                if attempt == max_attempts:
                    raise RuntimeError(
                        f"E3 VLM did not reach full coverage after {max_attempts} attempts"
                    )
                time.sleep(5)

        for phase in ("e3ground", "e3cal", "e3pol"):
            if not run_phase(original, phase) or not artifact(phase).exists():
                raise RuntimeError(f"{phase} failed or its result is missing")
        log("all E3 phases complete")
    finally:
        set_phase(original, original["phase"])


if __name__ == "__main__":
    main()
