"""Shared configuration loading. All scientific variants are committed config."""
import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(REPO_ROOT, "config")

with open(os.path.join(CONFIG_DIR, "shared.json")) as f:
    SHARED = json.load(f)
with open(os.path.join(CONFIG_DIR, "variant.json")) as f:
    VARIANT = json.load(f)

DATA_ROOT = os.path.expanduser(SHARED["data_root"])
CACHE_DIR = os.path.join(DATA_ROOT, SHARED["cache_subdir"])
RUNS_DIR = os.path.join(DATA_ROOT, SHARED["runs_subdir"])
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)

DATASETS = {
    "lift": {
        "raw": os.path.join(DATA_ROOT, "data", "lift_ph_raw.hdf5"),
        "lowdim": os.path.join(DATA_ROOT, "data", "lift_ph_lowdim.hdf5"),
    },
    "can": {
        "raw": os.path.join(DATA_ROOT, "data", "can_ph_raw.hdf5"),
        "lowdim": os.path.join(DATA_ROOT, "data", "can_ph_lowdim.hdf5"),
    },
}

TASK = VARIANT.get("task", "lift")


def cache_path():
    c = SHARED["feature_cache"]
    return os.path.join(
        CACHE_DIR, f"{TASK}_features_{c['version']}.h5"
    )


def run_dir():
    d = os.path.join(
        RUNS_DIR,
        f"{TASK}_{VARIANT['phase']}_{VARIANT.get('arm', 'none')}_seed{VARIANT.get('seed', 0)}",
    )
    os.makedirs(d, exist_ok=True)
    return d


def integrity_fingerprint():
    """Invalidate E0 receipts when source, shared config, or datasets change."""
    import hashlib
    from pathlib import Path
    from .data import sha256_file
    payload = {"shared": SHARED, "task": TASK,
               "source": {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
               "datasets": {k: sha256_file(v) for k, v in DATASETS[TASK].items()}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def require_integrity():
    path = os.path.join(CACHE_DIR, f"{TASK}_e0_receipt.json")
    with open(path) as f:
        receipt = json.load(f)
    if receipt.get("pass") is not True or receipt.get("fingerprint") != integrity_fingerprint():
        raise RuntimeError("E0 has not passed for the current code/config/data; rerun E0")
