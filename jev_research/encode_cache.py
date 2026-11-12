"""E0/E1 shared feature cache: replay-render agentview RGB through the frozen
framewise DINOv2 ViT-S/14 encoder; store per-frame pooled tokens.

Token layout per frame (float16, 6 x 384 = 2304 dims):
  [CLS, patch-grid 2x2 mean-pooled (4), global patch mean]
Frames are rendered by resetting the simulator to the recorded per-step state
(scene XML restored; fidelity still requires empirical validation).
The dynamics replay audit is run separately on a subset of episodes.
"""
import json
import os
import time

import h5py
import numpy as np

from . import config, data


def _encoder():
    import torch
    import timm

    name = config.SHARED["encoder"]["name"]
    m = timm.create_model(name, pretrained=True, num_classes=0,
                          dynamic_img_size=True)
    m.eval()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    m = m.to(dev)
    return m, dev


def _encode_batch(model, dev, frames_uint8):
    """frames_uint8: (B, 84, 84, 3) -> (B, 6, 384) pooled tokens, fp32."""
    import torch

    norm = config.SHARED["encoder"]["imagenet_norm"]
    mean = torch.tensor(norm["mean"], device=dev).view(1, 3, 1, 1)
    std = torch.tensor(norm["std"], device=dev).view(1, 3, 1, 1)
    x = torch.from_numpy(frames_uint8).to(dev).permute(0, 3, 1, 2).float() / 255.0
    x = (x - mean) / std
    with torch.no_grad():
        feats = model.forward_features(x)  # (B, 1+36, 384)
    cls = feats[:, 0]
    patches = feats[:, 1:]  # (B, 36, 384) on a 6x6 grid
    B = patches.shape[0]
    grid = patches.reshape(B, 6, 6, -1)
    pooled = []
    for i in (0, 3):
        for j in (0, 3):
            pooled.append(grid[:, i : i + 3, j : j + 3].mean(dim=(1, 2)))
    gmean = patches.mean(1)
    return torch.cat([cls.unsqueeze(1)] + [p.unsqueeze(1) for p in pooled] + [gmean.unsqueeze(1)], 1)


def reset_episode(env, group, state_index=0):
    """Restore the recorded scene XML as well as physics and controller state."""
    if "model_file" not in group.attrs:
        raise ValueError("Episode XML is required for faithful reconstruction")
    payload = {"model": group.attrs["model_file"],
               "states": group["states"][state_index]}
    if "ep_meta" in group.attrs:
        payload["ep_meta"] = group.attrs["ep_meta"]
    return env.reset_to(payload)


def close_env(env):
    if env is not None:
        target = getattr(env, "env", env)
        close = getattr(target, "close", None)
        if close:
            close()


def reset_controller_goals(env):
    """Synchronize controller targets after restoring a recorded physics state."""
    for robot in env.env.robots:
        robot.composite_controller.update_state()
        robot.composite_controller.reset()


def cache_contract():
    import hashlib
    import importlib.metadata
    payload = {"raw_sha256": data.sha256_file(config.DATASETS[config.TASK]["raw"]),
               "encoder": config.SHARED["encoder"],
               "features": config.SHARED["feature_cache"],
               "packages": {p: importlib.metadata.version(p) for p in
                            ("torch", "timm", "robomimic", "robosuite", "mujoco")},
               "render": "episode XML; recorded state; agentview 84x84; v2"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def build_cache(needed_demos=None, audit_subset=10, save_png_demos=2):
    from . import env_setup
    import robomimic.utils.env_utils as EnvUtils
    task = config.TASK
    allowed = set(sum([v for k, v in data.load_split(task).items() if k != "test"], []))
    needed = list(needed_demos if needed_demos is not None else sorted(allowed))
    if not set(needed) <= allowed:
        raise ValueError("Locked test rendering is forbidden during E0/E1")
    path, contract = config.cache_path(), cache_contract()
    t0, n_frames, env = time.time(), 0, None
    with h5py.File(config.DATASETS[task]["raw"], "r") as src, h5py.File(path, "a") as out:
        if len(out) and out.attrs.get("contract") != contract:
            raise ValueError("Stale cache contract; use a new committed cache version")
        out.attrs["contract"] = contract
        todo = [d for d in needed if d not in out or not out[d].attrs.get("complete", False)]
        if not todo:
            return {"encoded_episodes": 0, "frames": 0, "contract": contract}
        model, dev = _encoder()
        # Pin the actual pretrained weights in addition to the model name.
        import hashlib
        digest = hashlib.sha256()
        for key, value in sorted(model.state_dict().items()):
            digest.update(key.encode()); digest.update(value.detach().cpu().numpy().tobytes())
        weights = digest.hexdigest()
        if "weights_sha256" in out.attrs and out.attrs["weights_sha256"] != weights:
            raise ValueError("Encoder weight mismatch")
        out.attrs["weights_sha256"] = weights
        try:
            env = EnvUtils.create_env_for_data_processing(
                env_meta=json.loads(src["data"].attrs["env_args"]),
                camera_names=["agentview"], camera_height=84, camera_width=84,
                reward_shaping=False)
            for idx, d in enumerate(todo):
                group = src["data"][d]
                states = group["states"][:]
                frames = [reset_episode(env, group)["agentview_image"]]
                for state in states[1:]:
                    frames.append(env.reset_to({"states": state})["agentview_image"])
                frames = np.stack(frames)
                toks = np.concatenate([_encode_batch(model, dev, frames[i:i+64]).cpu().numpy()
                                       for i in range(0, len(frames), 64)]).astype(np.float16)
                if not np.isfinite(toks).all():
                    raise ValueError(f"Nonfinite encoder output: {d}")
                if d in out:
                    del out[d]  # only incomplete groups in this versioned derived cache
                grp = out.create_group(d)
                grp.create_dataset("tokens", data=toks, compression="gzip")
                grp.attrs["n_frames"] = len(frames)
                grp.attrs["complete"] = True
                out.flush()
                n_frames += len(frames)
                if idx < save_png_demos:
                    from PIL import Image
                    Image.fromarray(frames[0]).save(os.path.join(config.CACHE_DIR, f"v2_sanity_{d}.png"))
                if (idx + 1) % 10 == 0:
                    print(f"[cache] {idx+1}/{len(todo)} episodes", flush=True)
        finally:
            close_env(env)
    dt = time.time() - t0
    return {"encoded_episodes": len(todo), "frames": n_frames, "seconds": dt,
            "contract": contract, "weights_sha256": weights}


def replay_audit(demos, max_episodes=10):
    from . import env_setup
    import robomimic.utils.env_utils as EnvUtils
    env = None
    results, image_errors = [], []
    try:
        with h5py.File(config.DATASETS[config.TASK]["raw"], "r") as src:
            env = EnvUtils.create_env_for_data_processing(
                env_meta=json.loads(src["data"].attrs["env_args"]),
                camera_names=["agentview"], camera_height=84, camera_width=84,
                reward_shaping=False)
            for d in demos[:max_episodes]:
                group = src["data"][d]
                states, actions = group["states"][:], group["actions"][:]
                reset_episode(env, group)
                nq, nv = env.env.sim.model.nq, env.env.sim.model.nv
                if states.shape[1] != 1 + nq + nv:
                    raise ValueError("Unexpected flattened state layout")
                qpos_err, qvel_err = [], []
                one_step_qpos_err, one_step_qvel_err = [], []
                checkpoint = min(20, len(states)-1)
                for t in range(len(states)-1):
                    obs = env.step(actions[t])[0]
                    reached = env.get_state()["states"]
                    e = np.abs(reached-states[t+1])
                    qpos_err.append(float(e[1:1+nq].max()))
                    qvel_err.append(float(e[1+nq:1+nq+nv].max()))
                    if t+1 == checkpoint:
                        replay_image = obs["agentview_image"].copy()
                # The actual learning contract is H=1. Reset controller goals
                # to the restored achieved pose before every recorded action so
                # accumulated numerical/contact drift cannot masquerade as a
                # transition-alignment failure.
                for t in range(len(states)-1):
                    env.reset_to({"states": states[t]})
                    reset_controller_goals(env)
                    env.step(actions[t])
                    reached = env.get_state()["states"]
                    e = np.abs(reached-states[t+1])
                    one_step_qpos_err.append(float(e[1:1+nq].max()))
                    one_step_qvel_err.append(float(e[1+nq:1+nq+nv].max()))
                stored_image = env.reset_to({"states": states[checkpoint]})["agentview_image"]
                image_errors.append(float(np.abs(replay_image.astype(float)-stored_image.astype(float)).mean()))
                results.append({"demo": d, "steps": len(states)-1, "nq": nq, "nv": nv,
                                "qpos_max": max(qpos_err), "qpos_mean": float(np.mean(qpos_err)),
                                "qvel_max": max(qvel_err), "qvel_mean": float(np.mean(qvel_err)),
                                "one_step_qpos_max": max(one_step_qpos_err),
                                "one_step_qpos_mean": float(np.mean(one_step_qpos_err)),
                                "one_step_qvel_max": max(one_step_qvel_err),
                                "one_step_qvel_mean": float(np.mean(one_step_qvel_err))})
    finally:
        close_env(env)
    return {"episodes": results, "image_state_consistency_mae": max(image_errors),
            "image_mae_per_episode": image_errors}
