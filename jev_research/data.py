"""Dataset access, deterministic splits, and the label-access ledger.

Split policy (frozen in shared.json sizes, deterministic assignment):
  - episodes are sorted by numeric demo id, then assigned to partitions with a
    seeded permutation (split_seed). All renderings of an episode stay together.
  - labeled subsets are prefixes of each partition's order (nested, fixed).
The ledger enumerates exactly which episode ids may expose native actions to
which stage; every consumer of actions must go through these lists.
"""
import hashlib
import json
import os

import h5py
import numpy as np

from . import config


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def demo_num(name):
    return int(name.split("_")[-1])


def list_demos(task):
    path = config.DATASETS[task]["raw"]
    with h5py.File(path, "r") as f:
        demos = sorted(f["data"].keys(), key=demo_num)
        # v1.5 packaging: attrs.total counts transitions, not episodes
        total_transitions = int(f["data"].attrs["total"])
        assert total_transitions >= len(demos)
    return demos


def load_split(task):
    """Deterministic split: sort ids, seeded permutation, fixed slices."""
    sp = config.SHARED["split"]
    demos = list_demos(task)
    rng = np.random.default_rng(sp["split_seed"])
    perm = rng.permutation(len(demos))
    ordered = [demos[i] for i in perm]
    sizes = sp["sizes"]
    out = {}
    i = 0
    for name in ["train", "dev", "calib", "test"]:
        out[name] = ordered[i : i + sizes[name]]
        i += sizes[name]
    assert i == len(demos)
    return out


def pilot_subset(split):
    """Pilot (E1) scope: slices of the frozen split. Locked test untouched."""
    ps = config.SHARED["pilot"]
    return {
        "train_pool": split["train"][: ps["train_pool"]],
        "labeled_train": split["train"][: ps["labeled_train"]],
        "dev_labeled": split["dev"][: ps["dev_labeled"]],
        "calib": split["calib"][: ps["calib"]],
        "eval_holdout": split["dev"][ps["dev_labeled"] : ps["dev_labeled"] + ps["eval_holdout"]],
    }


def label_ledger(task, split=None):
    """Union of action-labeled episodes permitted upstream of final evaluation."""
    split = split or load_split(task)
    p = pilot_subset(split)
    arms_extra = {"ref": p["train_pool"]}
    ledger = {
        "annotator_train_labels": p["labeled_train"],
        "selection_labels": p["dev_labeled"],
        "calibration_labels": p["calib"],
        "consumer_gold_default": p["labeled_train"],
        "ref_extra_gold": [d for d in p["train_pool"] if d not in p["labeled_train"]],
        "eval_only": {
            "annotation_accuracy_and_pseudo_label_diagnostics": p["eval_holdout"],
            "notes": "evaluation-only; never available to training or selection",
        },
        "locked_test": {"episodes": len(split["test"]), "access": "none until E2"},
    }
    return ledger


def ledger_counts(ledger):
    up = set(ledger["annotator_train_labels"]) | set(ledger["selection_labels"]) | set(ledger["calibration_labels"])
    return {
        "labeled_episodes_upstream_of_eval": len(up),
        "breakdown": {
            "annotator_train": len(ledger["annotator_train_labels"]),
            "selection": len(ledger["selection_labels"]),
            "calibration": len(ledger["calibration_labels"]),
        },
    }


def ids_hash(ids):
    return hashlib.sha256(",".join(sorted(ids)).encode()).hexdigest()[:16]


def load_episode(task, demo):
    path = config.DATASETS[task]["raw"]
    with h5py.File(path, "r") as f:
        g = f["data"][demo]
        states = g["states"][:]
        actions = g["actions"][:]
    return states, actions


def load_lowdim_episodes(task, demos):
    path = config.DATASETS[task]["lowdim"]
    out = {}
    with h5py.File(path, "r") as f:
        for d in demos:
            g = f["data"][d]
            if "obs" not in g:
                continue
            obs = {}
            for k in g["obs"]:
                obs[k] = g["obs"][k][:]
            out[d] = {"obs": obs, "actions": g["actions"][:]}
    return out


def action_stats(task, labeled_ids):
    """Action normalization stats from TRAIN-LABELED episodes only, with floor."""
    acts = []
    for d in labeled_ids:
        _, a = load_episode(task, d)
        acts.append(a)
    acts = np.concatenate(acts, 0)
    std = acts.std(0)
    floor = config.SHARED["action_norm"]["std_floor"]
    std = np.maximum(std, floor)
    return {"mean": acts.mean(0).tolist(), "std": std.tolist()}


def write_split_manifest(task, split):
    path = os.path.join(config.CACHE_DIR, f"{task}_split_manifest.json")
    payload = {
        "task": task,
        "split": split,
        "ledger": label_ledger(task, split),
        "ledger_counts": ledger_counts(label_ledger(task, split)),
        "split_seed": config.SHARED["split"]["split_seed"],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=1)
    return path
