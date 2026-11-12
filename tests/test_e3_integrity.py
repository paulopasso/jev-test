"""E3 regression/integrity tests: schema parsers, z-encoding, leakage
guards, cache-key determinism, pseudo-bundle contracts. No network, no
simulator, no .env access."""
import hashlib
import json
import sys
import os
import tempfile
from unittest.mock import patch

import h5py

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jev_research import e3


def test_semantic_parse_and_encode():
    raw = json.dumps({
        "ee_motion": {"x": 1, "y": 0, "z": -1, "mag": "medium"},
        "rotation": {"roll": 0, "pitch": 1, "yaw": 0},
        "gripper_event": "close",
        "object_motion": "small",
        "contact_change": "made",
        "interaction": "grasp",
        "confidence": 0.8,
        "ambiguity": "low"})
    p = e3.parse_semantic(raw)
    assert len(p["z"]) == e3.Z_DIM == 24
    assert p["z"][:4] == [1.0, 0.0, -1.0, 2 / 3]
    assert p["z"][4:7] == [0.0, 1.0, 0.0]
    assert p["z"][7:10] == [0.0, 1.0, 0.0]
    assert p["z"][10] == 0.5
    assert p["z"][11:15] == [0.0, 1.0, 0.0, 0.0]
    assert p["z"][15:22] == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert p["z"][22] == 0.8 and p["z"][23] == 0.0
    print("semantic parse/encode OK")


def test_semantic_rejects():
    bad = [
        '{"ee_motion": {"x": 2, "y": 0, "z": 0, "mag": "small"}, "rotation": {"roll": 0, "pitch": 0, "yaw": 0}, "gripper_event": "none", "object_motion": "none", "contact_change": "none", "interaction": "other", "confidence": 0.5, "ambiguity": "low"}',
        '{"ee_motion": {"x": 0, "y": 0, "z": 0, "mag": "huge"}, "rotation": {"roll": 0, "pitch": 0, "yaw": 0}, "gripper_event": "none", "object_motion": "none", "contact_change": "none", "interaction": "other", "confidence": 0.5, "ambiguity": "low"}',
        '{"ee_motion": {"x": 0, "y": 0, "z": 0, "mag": "none"}, "rotation": {"roll": 0, "pitch": 0, "yaw": 0}, "gripper_event": "none", "object_motion": "none", "contact_change": "none", "interaction": "other", "confidence": 1.5, "ambiguity": "low"}',
        'not json at all',
    ]
    for b in bad:
        try:
            e3.parse_semantic(b)
        except Exception:
            continue
        raise AssertionError(f"should have rejected: {b[:60]}")
    print("semantic rejection OK")


def test_direct_parse():
    raw = json.dumps({"dx": 0.5, "dy": -0.2, "dz": 0.0, "droll": 0.1,
                      "dpitch": 0.0, "dyaw": -0.05, "gripper": 1,
                      "confidence": 0.7})
    p = e3.parse_direct(raw)
    assert p["a"] == [0.5, -0.2, 0.0, 0.1, 0.0, -0.05, 1.0]
    assert p["confidence"] == 0.7
    try:
        e3.parse_direct(json.dumps({"dx": 2.0, "dy": 0, "dz": 0, "droll": 0,
                                    "dpitch": 0, "dyaw": 0, "gripper": 1,
                                    "confidence": 0.5}))
    except ValueError:
        print("direct parse/range OK")
        return
    raise AssertionError("direct should reject out-of-range")


def test_context_edge_duplication():
    # 10 states -> transitions 0..8; edges duplicate nearest frame
    assert e3.context_idxs(0, 10, (-1, 0, 1, 2)) == [0, 0, 1, 2]
    assert e3.context_idxs(8, 10, (-1, 0, 1, 2)) == [7, 8, 9, 9]
    assert e3.context_idxs(4, 10, (0, 1)) == [4, 5]
    print("context edges OK")


def test_cache_key_determinism():
    from jev_research import vlm
    k1 = vlm.record_key("m", "v", "demo_1", [7], ["ab12", "cd34"])
    k2 = vlm.record_key("m", "v", "demo_1", [7], ["ab12", "cd34"])
    k3 = vlm.record_key("m", "v", "demo_1", [8], ["ab12", "cd34"])
    assert k1 == k2 and k1 != k3 and len(k1) == 64
    print("cache key OK")


def test_action_leak_guard():
    class FakeEps:
        acts = {"demo_117": np.zeros((5, 7))}
    try:
        e3.assert_no_action_leak(["demo_117", "demo_39"], FakeEps())
    except RuntimeError:
        print("leak guard OK")
        return
    raise AssertionError("leak guard should have fired")


def test_shuffled_z_deterministic():
    Z = np.arange(40, dtype=np.float32).reshape(10, 4)
    a = e3.shuffled_z(Z, seed=999)
    b = e3.shuffled_z(Z, seed=999)
    assert np.array_equal(a, b) and not np.array_equal(a, Z)
    print("shuffled-z OK")


def test_render_restores_episode_scene_before_states():
    """A reused simulator must not render a new demo in the previous scene."""
    class FakeEnv:
        def __init__(self):
            self.calls = []

        def reset_to(self, payload):
            self.calls.append(dict(payload))
            return {"agentview_image": np.zeros((2, 2, 3), dtype=np.uint8)}

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "raw.h5")
        with h5py.File(path, "w") as f:
            g = f.create_group("data/demo_1")
            g.attrs["model_file"] = "<mujoco model='demo_1'/>"
            g.create_dataset("states", data=np.arange(6).reshape(3, 2))
        env = FakeEnv()
        with patch.object(e3.config, "DATASETS", {e3.config.TASK: {"raw": path}}):
            images = e3.render_pngs("demo_1", [1, 2], size=2, env=env)
        assert len(images) == 2
        assert env.calls[0]["model"] == "<mujoco model='demo_1'/>"
        np.testing.assert_array_equal(env.calls[0]["states"], [2, 3])
        assert "model" not in env.calls[1]
        np.testing.assert_array_equal(env.calls[1]["states"], [4, 5])


def test_legacy_image_cache_is_isolated():
    from jev_research import vlm
    with tempfile.TemporaryDirectory() as tmp, patch.object(vlm.config, "DATA_ROOT", tmp):
        assert vlm.cache_path("semantic").endswith("semantic_v2_episode_xml.jsonl")
        assert "v2_episode_xml" in e3.SEMANTIC_VERSION


def test_failed_cache_record_is_retried_and_success_is_reused():
    from jev_research import vlm
    job = {"demo": "demo_1", "tids": [2], "frame_hashes": ["new-scene"],
           "frames": [b"unused"]}
    key = vlm.record_key("model", "v2", job["demo"], job["tids"],
                         job["frame_hashes"])
    with tempfile.TemporaryDirectory() as tmp, patch.object(vlm.config, "DATA_ROOT", tmp):
        path = vlm.cache_path("semantic")
        vlm.append_record(path, {"key": key, "status": "http_fail"})
        with patch.object(vlm, "load_key", return_value="test-key"), \
             patch.object(vlm, "chat", return_value=('{"ok": true}',
                                                       {"prompt_tokens": 10,
                                                        "completion_tokens": 2}, 1)) as chat:
            result = vlm.run_pass("semantic", "v2", "model", [job],
                                  lambda _: [], json.loads, 10, lambda _: None,
                                  workers=1)
            assert result["requests"] == 1 and result["ok_total"] == 1
            assert chat.call_count == 1
            again = vlm.run_pass("semantic", "v2", "model", [job],
                                 lambda _: [], json.loads, 10, lambda _: None,
                                 workers=1)
            assert again["requests"] == 0 and again["ok_total"] == 1
            assert chat.call_count == 1


if __name__ == "__main__":
    test_semantic_parse_and_encode()
    test_semantic_rejects()
    test_direct_parse()
    test_context_edge_duplication()
    test_cache_key_determinism()
    test_action_leak_guard()
    test_shuffled_z_deterministic()
    print("ALL E3 REGRESSION TESTS PASS")
