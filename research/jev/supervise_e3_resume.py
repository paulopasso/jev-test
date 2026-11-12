"""One-shot safety net for a previously launched E3 resume process.

Waits for the older resume PID to exit. If E3 is incomplete, relaunches the
current resume script once; if it completed, does nothing. Never runs a second
E3 pipeline concurrently with the older one.
"""

import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
RESULT = Path.home() / ".local/share/jev_test/runs/lift_e3pol_none_seed0/e3pol.json"


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def main():
    old_pid = int(sys.argv[1])
    print(f"[supervise_e3] waiting for {old_pid}", flush=True)
    while alive(old_pid):
        time.sleep(10)
    if RESULT.exists():
        print("[supervise_e3] E3 already complete", flush=True)
        return
    print("[supervise_e3] resuming incomplete E3", flush=True)
    result = subprocess.run([sys.executable, "-u", "research/jev/resume_e3.py"], cwd=ROOT)
    print(f"[supervise_e3] resumed pipeline exit={result.returncode}", flush=True)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
